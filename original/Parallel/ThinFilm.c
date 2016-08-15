/*
 * -----------------------------------------------------------------
 * 
 * $Date: 2016-03-02 (Wed) $
 * -----------------------------------------------------------------
 * Programmer(s): Ananthan M and G. Tomar (IISc Bangalore)  @ LLNL
 * -----------------------------------------------------------------
 *
 * Thin Film Equation Solver
 * dHdt = -(D/DX(H^3 HXXX - HX*(1 -(L0/H)**6)))
 * L0 = 0.02 Precurse film thickness
 * Computational Domain: 0 <= x <= 5* (2*Pi sqrt(2))
 * Time: Use a function to check if the change in morphology is small
 * and stop if it doesn't change for long? 0<=T<=1000
 * Spatial derivatives are approximated using finite differences and ODEs 
 * in time are are solved using CVODE
 *
 * (Taken from cvDiurnal_kry example) The problem is solved by 
 *  CVODE on NPE processors, treated
 * as a rectangular process grid of size NPEX by NPEY, with
 * NPE = NPEX*NPEY. Each processor contains a subgrid of size MXSUB
 * by MYSUB of the (x,y) mesh.  Thus the actual mesh sizes are
 * MX = MXSUB*NPEX and MY = MYSUB*NPEY, and the ODE system size is
 * neq = 2*MX*MY.
 *
 * The solution is done with the BDF/GMRES method (i.e. using the
 * CVSPGMR linear solver) and the block-diagonal part of the
 * Newton matrix as a left preconditioner. A copy of the
 * block-diagonal part of the Jacobian is saved and conditionally
 * reused within the preconditioner routine.
 *
 * Performance data and sampled solution values are printed at
 * selected output times, and all performance counters are printed
 * on completion.
 *
 * This version uses MPI for user routines.
 * 
 * Execution: mpirun -np N cvThinFilm_kry_p   with N = NPEX*NPEY
 * (see constants below).
 * -----------------------------------------------------------------
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cvode/cvode.h>               /* prototypes for CVODE fcts. */
#include <cvode/cvode_spgmr.h>         /* prototypes & constants for CVSPGMR  */
#include <nvector/nvector_parallel.h>  /* def. of N_Vector, macro NV_DATA_P  */
#include <sundials/sundials_dense.h>   /* prototypes for small dense fcts. */
#include <sundials/sundials_types.h>   /* definitions of realtype, booleantype */
#include <sundials/sundials_math.h>    /* definition of macros SUNSQR and EXP */

#include <mpi.h>                       /* MPI constants and types */

/* Problem Constants */

#define NVARS        1                    /* Film Thickness */
#define L0           0.02                 /* Precursor film thickness*/

#define T0           RCONST(0.0)          /* initial time */
#define NOUT         12                   /* number of output times */
#define PI         RCONST(3.1415926535898)  /* pi */ 

#define XMIN         RCONST(0.0)          /* grid boundaries in x  */
#define XMAX         RCONST(2.*2.8284*PI)           

#define NPEX         4              /* no. PEs in x direction of PE array */
#define MXSUB        5              /* no. x points per subgrid */
#define MX           (NPEX*MXSUB)   /* MX = number of x mesh points */
#define BPAD          2             /* offset boundary padding */

/* CVodeInit Constants */

#define RTOL    RCONST(1.0e-5)    /* scalar relative tolerance */
#define FLOOR   RCONST(100.0)     /* value of Film Thickness, H, from relative*/                                   /* to absolute tolerance */
#define ATOL    (RTOL*FLOOR)      /* scalar absolute tolerance */

/* Type : UserData 
   contains problem constants, preconditioner blocks, pivot arrays, 
   grid constants, and processor indices, as well as data needed
   for the preconditiner */

typedef struct {

  realtype lo6, dx;/*6th power of L0*/
  realtype uext[NVARS*(MXSUB+4)];
  int my_pe,isubx;
  int nvmxsub, nvmxsub4; /**/
  MPI_Comm comm;

  /* For preconditioner */
  realtype *P[MXSUB], *Jbd[MXSUB];
  long int *pivot[MXSUB];

} *UserData;

/* Private Helper Functions */
/*Initialization and destory subroutines*/
static void InitUserData(int my_pe, MPI_Comm comm, UserData data);
static void FreeUserData(UserData data);
static void SetInitialProfiles(N_Vector u, UserData data);

/*Print subroutines*/
static void PrintOutput(void *cvode_mem, int my_pe, MPI_Comm comm,
                        N_Vector u, realtype t);
static void PrintFinalStats(void *cvode_mem);

static void PrintData(realtype t, realtype umax, long int nst);
 /*MPI Communication subroutines*/
static void BSend(MPI_Comm comm, 
                  int my_pe,int isubx, 
                  realtype udata[]);
static void BRecvPost(MPI_Comm comm, MPI_Request request[], 
                      int my_pe, int isubx,
		      realtype uext[]);
static void BRecvWait(MPI_Request request[], int isubx,                     
                       realtype uext[]
                      );
static void ucomm(realtype t, N_Vector u, UserData data);

/*Calculation of udot: called from subroutine f*/
static void fcalc(realtype t, realtype udata[], realtype dudata[],
                  UserData data);

/* Functions Called by the Solver */

static int f(realtype t, N_Vector u, N_Vector udot, void *user_data);

static int Precond(realtype tn, N_Vector u, N_Vector fu,
                   booleantype jok, booleantype *jcurPtr, 
                   realtype gamma, void *user_data, 
                   N_Vector vtemp1, N_Vector vtemp2, N_Vector vtemp3);

static int PSolve(realtype tn, N_Vector u, N_Vector fu, 
                  N_Vector r, N_Vector z, 
                  realtype gamma, realtype delta,
                  int lr, void *user_data, N_Vector vtemp);


/* Private function to check function return values */

static int check_flag(void *flagvalue, char *funcname, int opt, int id);

/***************************** Main Program ******************************/

int main(int argc, char *argv[])
{ 
  realtype abstol, reltol, t, tout , umax, umin;
  N_Vector u;
  UserData data;
  void *cvode_mem;
  int iout, flag, my_pe, npes ;
  long int neq, local_N , nst;
  MPI_Comm comm;

  u = NULL;
  data = NULL;
  cvode_mem = NULL;

 /* Set problem size neq */
  neq = NVARS*MX;

  /* Get processor number and total number of pe's */
  MPI_Init(&argc, &argv);
  comm = MPI_COMM_WORLD;
  MPI_Comm_size(comm, &npes);
  MPI_Comm_rank(comm, &my_pe);

  if (npes != NPEX) {
    if (my_pe == 0)
      fprintf(stderr, "\nMPI_ERROR(0): npes = %d is not equal to NPEX = %d\n\n",
	      npes,NPEX);
    MPI_Finalize();
    return(1);
  }
  
  /* Set local length */
  local_N = NVARS*MXSUB;
  
  /* Allocate and load user data block; allocate preconditioner block */
  data = (UserData) malloc(sizeof *data);
  if (check_flag((void *)data, "malloc", 2, my_pe)) MPI_Abort(comm, 1);
  InitUserData(my_pe, comm, data);

  /* Allocate u, and set initial values and tolerances */ 
  u = N_VNew_Parallel(comm, local_N, neq);
  if (check_flag((void *)u, "N_VNew", 0, my_pe)) MPI_Abort(comm, 1);
  SetInitialProfiles(u, data);
  
  
  abstol = ATOL; reltol = RTOL;

  /* Call CVodeCreate to create the solver memory and specify the 
   * Backward Differentiation Formula and the use of a Newton iteration */
  cvode_mem = CVodeCreate(CV_BDF, CV_NEWTON);
  if (check_flag((void *)cvode_mem, "CVodeCreate", 0, my_pe)) MPI_Abort(comm, 1);

  /* Set the pointer to user-defined data */
  flag = CVodeSetUserData(cvode_mem, data);
  if (check_flag(&flag, "CVodeSetUserData", 1, my_pe)) MPI_Abort(comm, 1);

  /* Call CVodeInit to initialize the integrator memory and specify the
   * user's right hand side function in u'=f(t,u), the inital time T0, and
   * the initial dependent variable vector u. */
  flag = CVodeInit(cvode_mem, f, T0, u);
  if(check_flag(&flag, "CVodeInit", 1, my_pe)) return(1);

  /* Call CVodeSStolerances to specify the scalar relative tolerance
   * and scalar absolute tolerances */
  flag = CVodeSStolerances(cvode_mem, reltol, abstol);
  if (check_flag(&flag, "CVodeSStolerances", 1, my_pe)) return(1);

  /* Call CVSpgmr to specify the linear solver CVSPGMR 
     with left preconditioning and the maximum Krylov dimension maxl */
  flag = CVSpgmr(cvode_mem, PREC_LEFT, 0);
  if (check_flag(&flag, "CVSpgmr", 1, my_pe)) MPI_Abort(comm, 1);

  /* Set preconditioner setup and solve routines Precond and PSolve, 
     and the pointer to the user-defined block data */
  flag = CVSpilsSetPreconditioner(cvode_mem, Precond, PSolve);
  if (check_flag(&flag, "CVSpilsSetPreconditioner", 1, my_pe)) MPI_Abort(comm, 1);
  
  if (my_pe == 0)
    printf("\nThin Film Equation\n\n");
  
  umin = N_VMin(u);


  if (my_pe == 0)
    printf("\nmin u is %f\n\n",umin);


//  if (my_pe == 0)
//     N_VPrint_Parallel(u) ;

  /* In loop over output points, call CVode, print results, test for error */
  for (iout=1, tout = 0.0; iout <= NOUT; iout++) {
    flag = CVode(cvode_mem, tout, u, &t, CV_ONE_STEP);
    if (check_flag(&flag, "CVode", 1, my_pe)) break;

    //    PrintOutput(cvode_mem, my_pe, comm, u, t);
 
    umin = N_VMin(u);
    if (my_pe == 0)
       printf("\nmin u is %f\n\n",umin);
  
    flag = CVodeGetNumSteps(cvode_mem, &nst);
    check_flag(&flag, "CVodeGetNumSteps", 1, my_pe);
  }

  /* Print final statistics */  
  if (my_pe == 0) PrintFinalStats(cvode_mem);

  /* Free memory */
  N_VDestroy_Parallel(u);
  FreeUserData(data);
  CVodeFree(&cvode_mem);
  
  MPI_Finalize();
  
  return(0);
}

/***************** Functions Called by the Solver *************************/

/* f routine.  Evaluate f(t,y).  First call ucomm to do communication of 
   subgrid boundary data into uext.  Then calculate f by a call to fcalc. */

static int f(realtype t, N_Vector u, N_Vector udot, void *user_data)
{
  realtype *udata, *dudata;
  UserData data;

  udata = NV_DATA_P(u);
  dudata = NV_DATA_P(udot);
  data = (UserData) user_data;

  /* Call ucomm to do inter-processor communication */
  ucomm(t, u, data);

  /* Call fcalc to calculate all right-hand sides */
  fcalc(t, udata, dudata, data);

  return(0);
}

/*********************** Functions called by f ************************/
/* ucomm routine.  This routine performs all communication 
   between processors of data needed to calculate f. */

static void ucomm(realtype t, N_Vector u, UserData data)
{

  realtype *udata, *uext;
  MPI_Comm comm;
  int my_pe, isubx;
  long int nvmxsub;
  MPI_Request request[4];

  udata = NV_DATA_P(u);

  /* Get comm, my_pe, subgrid indices, data sizes, extended array uext */
  comm = data->comm;
  my_pe = data->my_pe;
  isubx = data->isubx;   
  nvmxsub = data->nvmxsub;
  
  uext = data->uext;

  /* Start receiving boundary data from neighboring PEs */
  BRecvPost(comm, request, my_pe, isubx,  uext);

  /* Send data from boundary of local grid to neighboring PEs */
  BSend(comm, my_pe, isubx,  udata);

  /* Finish receiving boundary data from neighboring PEs */
  BRecvWait(request, isubx,  uext);
}


/* Routine to start receiving boundary data from neighboring PEs.*/

static void BRecvPost(MPI_Comm comm, MPI_Request request[], 
                      int my_pe, int isubx, 
		      realtype uext[])
{


  /* If isubx = 0, receive data for left x-line of uext from last pe  */
  if (isubx == 0) {
    MPI_Irecv(&uext[0],BPAD, PVEC_REAL_MPI_TYPE,
                                         NPEX-1, 0, comm, &request[0]);
  }
  /* If isubx > 0, receive data for left x-line of uext (via bufleft) */
  if (isubx != 0) {
    MPI_Irecv(&uext[0], BPAD, PVEC_REAL_MPI_TYPE,
                                         my_pe-1, 0, comm, &request[1]);
  }


  /* If isubx =  NPEX-1, receive data for right x-line of uext (via bufright) */
  if (isubx == NPEX-1) {
    MPI_Irecv(&uext[NVARS*(MXSUB+2)], BPAD, PVEC_REAL_MPI_TYPE,
                                         0, 0, comm, &request[2]);
  }
  /* If isubx < NPEX-1, receive data for right x-line of uext (via bufright) */
  if (isubx != NPEX-1) {
    MPI_Irecv(&uext[NVARS*(MXSUB+2)], BPAD, PVEC_REAL_MPI_TYPE,
                                         my_pe+1, 0, comm, &request[3]);
  }
}


/* Routine to send boundary data to neighboring PEs */

static void BSend(MPI_Comm comm, 
                  int my_pe, int isubx,  
                  realtype udata[])
{
  int i, ly;

  /* If isubx = 0, send data from left y-line of u (via bufleft) */
  if (isubx == 0) {
    MPI_Send(&udata[0], BPAD, PVEC_REAL_MPI_TYPE, NPEX-1, 0, comm);   
  }
  /* If isubx > 0, send data from left y-line of u (via bufleft) */
  if (isubx != 0) {
    MPI_Send(&udata[0], BPAD, PVEC_REAL_MPI_TYPE, my_pe-1, 0, comm);   
  }
    
  /* If isubx = NPEX-1, send data from right y-line of u (via bufright) */
  if (isubx == NPEX-1) {
    MPI_Send(&udata[MXSUB-2], BPAD, PVEC_REAL_MPI_TYPE, 0, 0, comm);   
  }

  /* If isubx < NPEX-1, send data from right y-line of u (via bufright) */
  if (isubx != NPEX-1) {
    MPI_Send(&udata[MXSUB-2], BPAD, PVEC_REAL_MPI_TYPE, my_pe+1, 0, comm);   
  }
}



/* Routine to finish receiving boundary data from neighboring PEs.
   Notes:   */

static void BRecvWait(MPI_Request request[], 
                      int isubx, 
                      realtype uext[]
                      )
{
  MPI_Status status;



  /* If isubx = 0, receive data for left x-line of uext from last pe  */
  if (isubx == 0) {
    MPI_Wait(&request[0],&status);
  }
  /* If isubx > 0, receive data for left x-line of uext (via bufleft) */
  if (isubx != 0) {

    MPI_Wait(&request[1],&status);
  }


  /* If isubx =  NPEX-1, receive data for right x-line of uext (via bufright) */
  if (isubx == NPEX-1) {
  
    MPI_Wait(&request[2],&status);
  }
  /* If isubx < NPEX-1, receive data for right x-line of uext (via bufright) */
  if (isubx != NPEX-1) {
  
    MPI_Wait(&request[3],&status);
  }

}



/* fcalc routine. Compute f(t,y).  This routine assumes that communication 
   between processors of data needed to calculate f has already been done,
   and this data is in the work array uext. */

static void fcalc(realtype t, realtype udata[],
                  realtype dudata[], UserData data)
{
  realtype *uext;
  realtype delx,lo6,dx2,H1,H2,H3,H4,H5,H2o3,H3o3,H4o3,
           H2o6,H3o6,H4o6,P2c1,P2c2,P2,P3c1,P3c2,P3,
           P4c1,P4c2,P4,A,Ao3,C,Co3,B,D,Numerator;
  int i, ly;
  int isubx;
  long int nvmxsub, nvmxsub4;

  /* Get subgrid indices, data sizes, extended work array uext */
  isubx = data->isubx;   
  nvmxsub = data->nvmxsub; nvmxsub4 = data->nvmxsub4;
  uext = data->uext;

  /* Copy local segment of u vector into the working extended array uext */
  for (ly = 0; ly < MXSUB; ly++) {
     uext[ly+2] = udata[ly];
  }


  /* Make local copies of problem variables, for efficiency */
  delx = data->dx;
  lo6 = data->lo6;
  
  dx2 = delx*delx;

  /* Loop over all grid points in local subgrid */
  for (ly = 0; ly < MXSUB; ly++) {
  /* dudata[0] corresponds to dH3/dt  */  
    H1 = uext[ly] ;  
    H2 = uext[ly+1] ;  
    H3 = uext[ly+2] ;  
    H4 = uext[ly+3] ;  
    H5 = uext[ly+4] ;  
    
    H2o3 = pow(H2,3);
    H3o3 = pow(H3,3);
    H4o3 = pow(H4,3);
   
    H2o6 = pow(H2,6);
    H3o6 = pow(H3,6);
    H4o6 = pow(H4,6);
// we can do that this way or muliply H2o3*H203

    P2c1 = (H3 -2*H2 + H1)/dx2;
    P2c2 = (1 - lo6/H2o6)/(3*H2o3); 
    P2 = P2c1 - P2c2 ;


    P3c1 = (H4 -2*H3 + H2)/dx2;
    P3c2 = (1 - lo6/H3o6)/(3*H3o3) ;
    P3 = P3c1 - P3c2 ;
    
    
    P4c1 = (H5 -2*H4 + H3)/dx2;
    P4c2 = (1 - lo6/H4o6)/(3*H4o3) ;
    P4 = P4c1 - P4c2 ;

    
    A = (H4+H3)/2;
    Ao3 = pow(A,3);
    C = (H3+H2)/2;
    Co3 = pow(C,3);

    B = (P4 - P3)/delx;

    D = (P3 -P2)/delx;

    Numerator =  (Ao3*B) - (Co3*D);

    dudata[ly] = (-1*Numerator)/delx;



  }
}

/*********************** Private Helper Functions ************************/


/* Load constants in data */

static void InitUserData(int my_pe, MPI_Comm comm, UserData data)
{
  int lx,isubx;

  /* Set problem constants */
  data->lo6 = pow(L0,6);
  data->dx = (XMAX-XMIN)/((realtype)(MX-1));

  /* Set machine-related constants */
  data->comm = comm;
  data->my_pe = my_pe;

  /* isubx are the PE grid indices corresponding to my_pe */
//  isuby = my_pe/NPEX;
//  isubx = my_pe - isuby*NPEX;
  isubx = my_pe;
  data->isubx = isubx;
//  data->isuby = isuby;

  /* Set the sizes of a boundary x-line in u and uext */
  data->nvmxsub = NVARS*MXSUB;
  data->nvmxsub4 = NVARS*(MXSUB+4);

  /* Preconditioner-related fields */
  for (lx = 0; lx < MXSUB; lx++) {
      (data->P)[lx] = newRealArray(NVARS);
      (data->Jbd)[lx] = newRealArray(NVARS);
      (data->pivot)[lx] = newLintArray(NVARS);
  }
}

/* Free user data memory */

static void FreeUserData(UserData data)
{
  int lx;

  for (lx = 0; lx < MXSUB; lx++) {
      destroyArray((data->P)[lx]);
      destroyArray((data->Jbd)[lx]);
      destroyArray((data->pivot)[lx]);
  }

  free(data);
}

/* Print data */

static void PrintData(realtype t, realtype umax, long int nst)
{

#if defined(SUNDIALS_EXTENDED_PRECISION)
  printf("At t = %4.2Lf  max.norm(u) =%14.6Le  nst =%4ld \n", t, umax, nst);
#elif defined(SUNDIALS_DOUBLE_PRECISION)
  printf("At t = %4.2f  max.norm(u) =%14.6e  nst =%4ld \n", t, umax, nst);
#else
  printf("At t = %4.2f  max.norm(u) =%14.6e  nst =%4ld \n", t, umax, nst);
#endif

  return;
}

/* Set initial conditions in u */

static void SetInitialProfiles(N_Vector u, UserData data)
{
  int isubx,  lx, jx;
  realtype dx, x;
  realtype *udata;

  /* Set pointer to data array in vector u */
  udata = NV_DATA_P(u);

  /* Get mesh spacings, and subgrid indices for this PE */
  dx = data->dx;        
  isubx = data->isubx;  

  /* Load initial profiles of  local u vector.
  Here lx  are local mesh point indices on the local subgrid,
  and jx  are the global mesh point indices. */
    for (lx = 0; lx < MXSUB; lx++) {
      jx = lx + isubx*MXSUB;
      x = XMIN + jx*dx;

//      udata[lx] = 10 -  x;
      udata[lx] =  1 + 0.01*(rand()%10);
    }
  
}


/* Print final statistics contained in iopt */

static void PrintFinalStats(void *cvode_mem)
{
  long int lenrw, leniw ;
  long int lenrwLS, leniwLS;
  long int nst, nfe, nsetups, nni, ncfn, netf;
  long int nli, npe, nps, ncfl, nfeLS;
  int flag;

  flag = CVodeGetWorkSpace(cvode_mem, &lenrw, &leniw);
  check_flag(&flag, "CVodeGetWorkSpace", 1, 0);
  flag = CVodeGetNumSteps(cvode_mem, &nst);
  check_flag(&flag, "CVodeGetNumSteps", 1, 0);
  flag = CVodeGetNumRhsEvals(cvode_mem, &nfe);
  check_flag(&flag, "CVodeGetNumRhsEvals", 1, 0);
  flag = CVodeGetNumLinSolvSetups(cvode_mem, &nsetups);
  check_flag(&flag, "CVodeGetNumLinSolvSetups", 1, 0);
  flag = CVodeGetNumErrTestFails(cvode_mem, &netf);
  check_flag(&flag, "CVodeGetNumErrTestFails", 1, 0);
  flag = CVodeGetNumNonlinSolvIters(cvode_mem, &nni);
  check_flag(&flag, "CVodeGetNumNonlinSolvIters", 1, 0);
  flag = CVodeGetNumNonlinSolvConvFails(cvode_mem, &ncfn);
  check_flag(&flag, "CVodeGetNumNonlinSolvConvFails", 1, 0);

  flag = CVSpilsGetWorkSpace(cvode_mem, &lenrwLS, &leniwLS);
  check_flag(&flag, "CVSpilsGetWorkSpace", 1, 0);
  flag = CVSpilsGetNumLinIters(cvode_mem, &nli);
  check_flag(&flag, "CVSpilsGetNumLinIters", 1, 0);
  flag = CVSpilsGetNumPrecEvals(cvode_mem, &npe);
  check_flag(&flag, "CVSpilsGetNumPrecEvals", 1, 0);
  flag = CVSpilsGetNumPrecSolves(cvode_mem, &nps);
  check_flag(&flag, "CVSpilsGetNumPrecSolves", 1, 0);
  flag = CVSpilsGetNumConvFails(cvode_mem, &ncfl);
  check_flag(&flag, "CVSpilsGetNumConvFails", 1, 0);
  flag = CVSpilsGetNumRhsEvals(cvode_mem, &nfeLS);
  check_flag(&flag, "CVSpilsGetNumRhsEvals", 1, 0);

  printf("\nFinal Statistics: \n\n");
  printf("lenrw   = %5ld     leniw   = %5ld\n", lenrw, leniw);
  printf("lenrwls = %5ld     leniwls = %5ld\n", lenrwLS, leniwLS);
  printf("nst     = %5ld\n"                  , nst);
  printf("nfe     = %5ld     nfels   = %5ld\n"  , nfe, nfeLS);
  printf("nni     = %5ld     nli     = %5ld\n"  , nni, nli);
  printf("nsetups = %5ld     netf    = %5ld\n"  , nsetups, netf);
  printf("npe     = %5ld     nps     = %5ld\n"  , npe, nps);
  printf("ncfn    = %5ld     ncfl    = %5ld\n\n", ncfn, ncfl); 
}
/*********************** Private Helper Function ************************/

/* Check function return value...
     opt == 0 means SUNDIALS function allocates memory so check if
              returned NULL pointer
     opt == 1 means SUNDIALS function returns a flag so check if
              flag >= 0
     opt == 2 means function allocates memory so check if returned
              NULL pointer */

static int check_flag(void *flagvalue, char *funcname, int opt, int id)
{
  int *errflag;

  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
  if (opt == 0 && flagvalue == NULL) {
    fprintf(stderr, "\nSUNDIALS_ERROR(%d): %s() failed - returned NULL pointer\n\n",
	    id, funcname);
    return(1); }

  /* Check if flag < 0 */
  else if (opt == 1) {
    errflag = (int *) flagvalue;
    if (*errflag < 0) {
      fprintf(stderr, "\nSUNDIALS_ERROR(%d): %s() failed with flag = %d\n\n",
	      id, funcname, *errflag);
      return(1); }}

  /* Check if function returned NULL pointer - no memory allocated */
  else if (opt == 2 && flagvalue == NULL) {
    fprintf(stderr, "\nMEMORY_ERROR(%d): %s() failed - returned NULL pointer\n\n",
	    id, funcname);
    return(1); }

  return(0);
}

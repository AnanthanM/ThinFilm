/*
 * -----------------------------------------------------------------
 * 
 * $Date: 2016-03-02 (Wed) $
 * -----------------------------------------------------------------
 * Programmer(s): Ananthan M and G. Tomar (IISc Bangalore)  
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


#define ZERO RCONST(0.0)
#define ONE  RCONST(1.0)
#define TWO  RCONST(2.0)

#define NVARS        1                    /* Film Thickness */
#define L0           RCONST(0.02)                 /* Precursor film thickness*/
#define T0           ZERO                /* initial time */
#define NOUT         12000000                   /* number of output times */
#define PI         RCONST(3.1415926535898)  /* pi */ 

#define XMIN         RCONST(0.0)          /* grid boundaries in x  */
#define XMAX         RCONST(1.*8.885765)           

#define NPEX         2              /* no. PEs in x direction of PE array */
#define MXSUB        60              /* no. x points per subgrid */
#define MX           (NPEX*MXSUB)   /* MX = number of x mesh points */
#define NEQ           MX  
#define BPAD          2             /* offset boundary padding */

/* CVodeInit Constants */

#define RTOL    RCONST(1.0e-9)    /* scalar relative tolerance */
#define FLOOR   RCONST(100.0)     /* value of Film Thickness, H, from relative*/                                   /* to absolute tolerance */
#define ATOL    (RTOL*FLOOR)      /* scalar absolute tolerance */

/* Type : UserData 
   contains problem constants, preconditioner blocks, pivot arrays, 
   grid constants, and processor indices, as well as data needed
   for the preconditiner */

typedef struct {

  realtype lo_6, dx;/*6th power of L0*/
  realtype uext[NVARS*(MXSUB+4)];
  int my_pe,isubx;
  int nvmxsub, nvmxsub4; /**/
  MPI_Comm comm;

  /* For preconditioner */
  N_Vector pp,Jac;
} *UserData;

/* Private Helper Functions */
/*Initialization and outines*/
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
  long int neq, local_N,nst, mudq, mldq, mukeep, mlkeep;
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
  


  if (my_pe == 0)
     N_VPrint_Parallel(u) ;

  /* In loop over output points, call CVode, print results, test for error */
  for (iout=1, tout = 10.0; iout <= NOUT; iout++) {
    flag = CVode(cvode_mem, tout, u, &t, CV_ONE_STEP);
    if (check_flag(&flag, "CVode", 1, my_pe)) break;

    //    PrintOutput(cvode_mem, my_pe, comm, u, t);
 
    umin = N_VMin(u);
    if (my_pe == 0)
       printf("\nmin u is %f at time %f \n\n ",umin,t);
  
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
//  ucomm(t, u, data);

  /* Call fcalc to calculate all right-hand sides */
  fcalc(t, udata, dudata, data);

  return(0);
}


/* Preconditioner setup routine. Generate and preprocess P. */

static int Precond(realtype tn, N_Vector u, N_Vector fu,
                   booleantype jok, booleantype *jcurPtr, realtype gamma,
                   void *user_data, N_Vector vtemp1, N_Vector vtemp2,
                   N_Vector vtemp3)
{ 
  realtype dx,lo_6,dx2,
           Hi,Him1,Hip1,Him2,Hip2,
           Him1_3,Hi_3,Hip1_3,
           Hi_4,Hi_10,
           Him1_6,Hi_6,Hip1_6,
           HipHip1,HipHim1,
           HipHip1_2,HipHim1_2,
           HipHip1_3,HipHim1_3,
           D_P_i,D_P_im1,D_P_ip1,
           A,B,K; //A and B are differences in disjoining pressures
  
 // realtype uperiodic[MXSUB+4];
  realtype *uext;

  int jx;
  realtype *udata, *ppv, *Jacv;
  UserData data;
  
  /* Make local copies of pointers in user_data, and of pointer to u's data */
  
  data = (UserData) user_data;
  uext = data->uext;
  
  ppv = NV_DATA_P(data->pp);
  Jacv = NV_DATA_P(data->Jac);
  
  N_VConst(ONE,data->pp);
  N_VConst(ONE,data->Jac);

  udata = NV_DATA_P(u);
  
  if (jok) {
    
    /* jok = TRUE: Copy Jbd to P */
    
    for(jx = 0;jx<MX;jx++)
      ppv[jx] = Jacv[jx];
    
    *jcurPtr = FALSE;
    
  }
  
  else {
    /* jok = FALSE: Generate Jbd from scratch and copy to P */
    
    /* Make local copies of problem variables, for efficiency. */
    
    dx = data->dx;
    lo_6 = data->lo_6;
    dx2 = dx*dx;
    
    /* Call ucomm to do inter-processor communication */
//    ucomm(tn, u, data);
    
    for(jx = 0;jx<MXSUB;jx++)
    {
      uext[jx + 2] = udata[jx];
    }
    uext[0] = uext[MXSUB];
    uext[1] = uext[MXSUB+1];
    uext[MXSUB+2] = uext[2];
    uext[MXSUB+3] = uext[3];

    for(jx =0;jx<MXSUB;jx++){
           
/*      Hi   =  uperiodic[jx+2];
      Him1 =  uperiodic[jx+1];
      Him2 =  uperiodic[jx];
      Hip1 =  uperiodic[jx+3];
      Hip2 =  uperiodic[jx+4];
 */     
      Hi   =  uext[jx+2];
      Him1 =  uext[jx+1];
      Him2 =  uext[jx];
      Hip1 =  uext[jx+3];
      Hip2 =  uext[jx+4];
 
      Him1_3 = pow(Him1,3);
      Hi_3 = pow(Hi,3);
      Hip1_3 = pow(Hip1,3);
    
      Hi_4 = pow(Hi,4);
      Hi_10 = pow(Hi,10);

      Him1_6 = pow(Him1,6);
      Hi_6 = pow(Hi,6);
      Hip1_6 = pow(Hip1,6);
    
      HipHip1 = Hi + Hip1;
      HipHim1 = Hi + Him1;

      HipHip1_2 = HipHip1*HipHip1;
      HipHim1_2 = HipHim1*HipHim1;

      HipHip1_3 = HipHip1_2*HipHip1;
      HipHim1_3 = HipHim1_2*HipHim1;

      
      D_P_im1 = ((Hi -2*Him1 + Him2)/dx2) - ( (1 - lo_6/Him1_6)/(3*Him1_3)); 

      D_P_i  = ((Hip1 -2*Hi + Him1)/dx2) - ( (1 - lo_6/Hi_6)/(3*Hi_3) );
    
      D_P_ip1  = ((Hip2 -2*Hip1 + Hi)/dx2) - ( (1 - lo_6/Hip1_6)/(3*Hip1_3)) ;
     
      A = D_P_ip1 - D_P_i;
      B = D_P_i - D_P_im1;

      K = ((3/dx2)-(1/Hi_4)+(3*lo_6/Hi_10));

      Jacv[jx] = -((HipHip1_3*K) + (3*A*HipHip1_2) + (HipHim1_3*K) - (3*B*HipHim1_2))/(8*dx2); 
      
      ppv[jx] = Jacv[jx];

    }

    


    *jcurPtr = TRUE;
    
  }
  
  /* Scale by -gamma */
  
  for(jx=0;jx<MX;jx++){
    
    ppv[jx] =1- gamma* ppv[jx];

    ppv[jx] = ONE/ppv[jx];

  }
  

  return(0);
}

/* Preconditioner solve routine */

static int PSolve(realtype tn, N_Vector u, N_Vector fu,
                  N_Vector r, N_Vector z,
                  realtype gamma, realtype delta,
                  int lr, void *user_data, N_Vector vtemp)
{
  UserData data;
  data = (UserData) user_data;
  N_VProd(data->pp,r,z);
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
  
  realtype dx,lo_6,dx2,
           Hi,Him1,Hip1,Him2,Hip2,
           Him1_3,Hi_3,Hip1_3,
           Him1_6,Hi_6,Hip1_6,
           D_P_im1,D_P_i,D_P_ip1,
           Hip0p5,Him0p5,
           Hip0p5_3,Him0p5_3,
           dD_P_ip0p5,dD_P_im0p5;
  int i, ly,lx;
  int isubx;
  long int nvmxsub, nvmxsub4;

  /* Get subgrid indices, data sizes, extended work array uext */
  isubx = data->isubx;   
  nvmxsub = data->nvmxsub;
  nvmxsub4 = data->nvmxsub4;
  uext = data->uext;

  /* Copy local segment of u vector into the working extended array uext */
  for (ly = 0; ly < MXSUB; ly++) {
     uext[ly+2] = udata[ly];
  }

    uext[0] = uext[MXSUB];
    uext[1] = uext[MXSUB+1];
    uext[MXSUB+2] = uext[2];
    uext[MXSUB+3] = uext[3];

  /* Make local copies of problem variables, for efficiency */
  dx = data->dx;
  lo_6 = data->lo_6;
  
  dx2 = dx*dx;

  /* Loop over all grid points in local subgrid */
  
  for(lx = 0;lx<MXSUB;lx++)
  {
    Hi   =  uext[lx+2];
    Him1 =  uext[lx+1];
    Him2 =  uext[lx];
    Hip1 =  uext[lx+3];
    Hip2 =  uext[lx+4];


    Him1_3 = pow(Him1,3);
    Hi_3 = pow(Hi,3);
    Hip1_3 = pow(Hip1,3);
   
    Him1_6 = pow(Him1,6);
    Hi_6 = pow(Hi,6);
    Hip1_6 = pow(Hip1,6);
// we can do that this way or muliply Him1_3*Him103

    D_P_im1 = ((Hi -2*Him1 + Him2)/dx2) - ( (1 - lo_6/Him1_6)/(3*Him1_3)); 

    D_P_i  = ((Hip1 -2*Hi + Him1)/dx2) - ( (1 - lo_6/Hi_6)/(3*Hi_3) );
    
    D_P_ip1  = ((Hip2 -2*Hip1 + Hi)/dx2) - ( (1 - lo_6/Hip1_6)/(3*Hip1_3)) ;
   

    
    Hip0p5 = (Hip1+Hi)/2;
    Hip0p5_3 = pow(Hip0p5,3);
    Him0p5 = (Hi+Him1)/2;
    Him0p5_3 = pow(Him0p5,3);

    dD_P_ip0p5 = (D_P_ip1 - D_P_i)/dx;

    dD_P_im0p5 = (D_P_i - D_P_im1)/dx;


    dudata[lx] = -((Hip0p5_3*dD_P_ip0p5) - (Him0p5_3*dD_P_im0p5) )/dx ;
  }

}




/*********************** Private Helper Functions ************************/


/* Load constants in data */

static void InitUserData(int my_pe, MPI_Comm comm, UserData data)
{
  int isubx,local_N;

  /* Set problem constants */
  data->lo_6 = pow(L0,6);
  data->dx = (XMAX-XMIN)/((realtype)(MX-1));

  /* Set machine-related constants */
  data->comm = comm;
  data->my_pe = my_pe;

  /* isubx are the PE grid indices corresponding to my_pe */
  isubx = my_pe;
  data->isubx = isubx;

  /* Set the sizes of a boundary x-line in u and uext */
  data->nvmxsub = NVARS*MXSUB;
  data->nvmxsub4 = NVARS*(MXSUB+4);

  local_N = NVARS*MXSUB;
  /* Preconditioner-related fields */
  
  data->pp = NULL;
  /* A N-Vector to hold preconditioner diagonal elements*/
  data->pp = N_VNew_Parallel(comm,local_N,NEQ);
  if (check_flag((void *)data->pp, "N_VNew", 0, my_pe)) MPI_Abort(comm, 1);


  data->Jac = NULL;
  /* A N-Vector to hold Jacobian  diagonal elements*/
  data->Jac = N_VNew_Parallel(comm,local_N,NEQ);
  if (check_flag((void *)data->Jac, "N_VNew", 0, my_pe)) MPI_Abort(comm, 1);
}

/* Free user data memory */

static void FreeUserData(UserData data)
{
  N_VDestroy_Parallel(data->pp);
  N_VDestroy_Parallel(data->Jac);
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
 //     udata[lx] =  1 + 0.01*(rand()%10);
      udata[lx]  = 1.0 + 0.01*(cos( (jx*dx*2*PI)/XMAX )) ;
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

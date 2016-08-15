
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cvode/cvode.h>              /* main integrator header file */
#include <cvode/cvode_spgmr.h>        /* prototypes & constants for CVSPGMR */
#include <nvector/nvector_serial.h>   /* serial N_Vector types, fct., macros */
#include <sundials/sundials_band.h>  /* use generic band solver in precond. */
#include <sundials/sundials_types.h>  /* definition of realtype */
#include <sundials/sundials_math.h>   /* contains the macros ABS, SUNSQR, EXP */

/* Problem Constants */

#define ZERO RCONST(0.0)
#define ONE  RCONST(1.0)
#define TWO  RCONST(2.0)


#define T0           ZERO                 /* initial time */
#define NOUT         100000                  /* number of output times */
#define PI       RCONST(3.1415926535898)  /* pi */ 
#define L0       RCONST(0.02)  /* pi */ 

#define XMIN         ZERO                 /* grid boundaries in x  */
#define XMAX         RCONST(8.8857)           

#define MX           120             /* MX = number of x mesh points */

/* CVodeInit Constants */

#define RTOL    RCONST(1.0e-11)    /* scalar relative tolerance */
#define FLOOR   RCONST(100.0)     /* value of H  at which tolerances */
                                  /* change from relative to absolute      */
#define ATOL    (RTOL*FLOOR)      /* scalar absolute tolerance */
#define NEQ     (MX)  /* NEQ = number of equations */


/* Type : UserData 
   contains preconditioner blocks, pivot arrays, and problem constants */

typedef struct {
//  realtype  **P,**Jac;
  DlsMat P,Jac;
  long int *pivot;
  realtype  dx,lo_6;
}*UserData;

/* Private Helper Functions */

static UserData AllocUserData(void);
static void InitUserData(UserData data);
static void FreeUserData(UserData data);
static void SetInitialProfiles(N_Vector u, realtype dx);
static void PrintOutput(void *cvode_mem, N_Vector u, realtype t, FILE *fp);
static void PrintFinalStats(void *cvode_mem);
static int check_flag(void *flagvalue, char *funcname, int opt);

/* Functions Called by the Solver */

static int f(realtype t, N_Vector u, N_Vector udot, void *user_data);


static int Precond(realtype tn, N_Vector u, N_Vector fu,
                   booleantype jok, booleantype *jcurPtr, realtype gamma,
                   void *user_data, N_Vector vtemp1, N_Vector vtemp2,
                   N_Vector vtemp3);

static int PSolve(realtype tn, N_Vector u, N_Vector fu,
                  N_Vector r, N_Vector z,
                  realtype gamma, realtype delta,
                  int lr, void *user_data, N_Vector vtemp);


/*
 *-------------------------------
 * Main Program
 *-------------------------------
 */
int main()
{
  realtype abstol, reltol, t, tout,umin;
  N_Vector u;
  UserData data;
  void *cvode_mem;
  int iout, flag;

  FILE *fpGRT;
  fpGRT=fopen("growth.dat","w");
  u = NULL;
  data = NULL;
  cvode_mem = NULL;

  /* Allocate memory, and set problem data, initial values, tolerances */ 
  u = N_VNew_Serial(NEQ);
  if(check_flag((void *)u, "N_VNew_Serial", 0)) return(1);
  data = AllocUserData();
  if(check_flag((void *)data, "AllocUserData", 2)) return(1);
  InitUserData(data);
  SetInitialProfiles(u, data->dx);
  abstol=ATOL; 
  reltol=RTOL;

  /* Call CVodeCreate to create the solver memory and specify the 
   * Backward Differentiation Formula and the use of a Newton iteration */
  cvode_mem = CVodeCreate(CV_BDF, CV_NEWTON);
  if(check_flag((void *)cvode_mem, "CVodeCreate", 0)) return(1);

  /* Set the pointer to user-defined data */
  flag = CVodeSetUserData(cvode_mem, data);
  if(check_flag(&flag, "CVodeSetUserData", 1)) return(1);

  /* Call CVodeInit to initialize the integrator memory and specify the
   * user's right hand side function in u'=f(t,u), the inital time T0, and
   * the initial dependent variable vector u. */
  flag = CVodeInit(cvode_mem, f, T0, u);
  if(check_flag(&flag, "CVodeInit", 1)) return(1);

  /* Call CVodeSStolerances to specify the scalar relative tolerance
   * and scalar absolute tolerances */
  flag = CVodeSStolerances(cvode_mem, reltol, abstol);
  if (check_flag(&flag, "CVodeSStolerances", 1)) return(1);

  /* Call CVSpgmr to specify the linear solver CVSPGMR 
   * with left preconditioning and the maximum Krylov dimension maxl */
  flag = CVSpgmr(cvode_mem, PREC_LEFT, 0);
  if(check_flag(&flag, "CVSpgmr", 1)) return(1);

  /* Set the preconditioner solve and setup functions */
  flag = CVSpilsSetPreconditioner(cvode_mem, Precond, PSolve);
  if(check_flag(&flag, "CVSpilsSetPreconditioner", 1)) return(1);

  /* In loop over output points, call CVode, print results, test for error */
  printf(" \n Thin Film Equation \n\n");

  
   PrintOutput(cvode_mem, u, t,fpGRT);
//   FileOutput

  for (iout=1, tout = 10.0; iout <= NOUT; iout++) {
    
    flag = CVode(cvode_mem, tout, u, &t, CV_ONE_STEP);
    umin = N_VMin(u);
    if(umin < ZERO){ 
      printf("\n ERROR\n");
      break;
    }

    if((umin - 0.02) < 0.001){
      printf("The Born repulsion starts at time %lf",t);
      break;
    }

    PrintOutput(cvode_mem, u, t,fpGRT);
    if(check_flag(&flag, "CVode", 1)) break;
  }

  PrintFinalStats(cvode_mem);

  /* Free memory */
  N_VDestroy_Serial(u);
  FreeUserData(data);
  CVodeFree(&cvode_mem);

  fclose(fpGRT);
  return(0);
}

/*
 *-------------------------------
 * Private helper functions
 *-------------------------------
 */

/* Allocate memory for data structure of type UserData */

static UserData AllocUserData(void)
{
  UserData data;

  data = (UserData) malloc(sizeof *data);
  
  data->P = NewBandMat(MX,2,2,MX-1);
  data->Jac = NewBandMat(MX,2,2,MX-1); 
  data->pivot = newLintArray(MX);

  return(data);
}

/* Load problem constants in data */

static void InitUserData(UserData data)
{
  data->lo_6 = pow(L0,6);
  data->dx = (XMAX-XMIN)/((realtype)(MX-1));
}

/* Free data memory */

static void FreeUserData(UserData data)
{
  DestroyMat(data->P);
  DestroyMat(data->Jac);
  destroyArray(data->pivot);

  free(data);
}

/* Set initial conditions in u */

static void SetInitialProfiles(N_Vector u, realtype dx)
{
  int jx;
  realtype *udata;

  /* Set pointer to data array in vector u. */

  udata = NV_DATA_S(u);

  /* Load initial profiles of c1 and c2 into u vector */
   
  for(jx = 0;jx < MX; jx++){
//    udata[jx] = 1 + 0.01*(rand()%10);
    udata[jx] = 1 + 0.01*(cos( (jx*dx*2*PI)/XMAX  ));
//    udata[jx] = sin(jx*dx);
  }


}

/* Print current t, step count, order, stepsize, and sampled c1,c2 values */

static void PrintOutput(void *cvode_mem, N_Vector u, realtype t, FILE *fpGRT)
{
  int i;
  realtype umin, *udata,H0,H;
  realtype A,sigma,curve1,curve2;
  udata = NV_DATA_S(u);
  
  H0 = 1.0; 
  H = udata[0];

  A = H - H0;

  curve1 = log(A/0.010);

  curve2 = 0.25*t;

  fprintf(fpGRT,"%lf %lf %lf\n",t,curve1,curve2);
  printf(" \n At time %f curve1 is %f and LSA  is %f  ",t,curve1,curve2);

  umin = N_VMin(u);
 
/*
  printf("\n \n At time %f H is \n \n",t);
    
  for(i = 0;i<MX;i++){
    printf("\t %f \t",udata[i]);
  }
*/
  printf("\n \n At time %f H minimum is %f \n \n",t,umin);

}

/* Get and print final statistics */

static void PrintFinalStats(void *cvode_mem)
{
  long int lenrw, leniw ;
  long int lenrwLS, leniwLS;
  long int nst, nfe, nsetups, nni, ncfn, netf;
  long int nli, npe, nps, ncfl, nfeLS;
  int flag;

  flag = CVodeGetWorkSpace(cvode_mem, &lenrw, &leniw);
  check_flag(&flag, "CVodeGetWorkSpace", 1);
  flag = CVodeGetNumSteps(cvode_mem, &nst);
  check_flag(&flag, "CVodeGetNumSteps", 1);
  flag = CVodeGetNumRhsEvals(cvode_mem, &nfe);
  check_flag(&flag, "CVodeGetNumRhsEvals", 1);
  flag = CVodeGetNumLinSolvSetups(cvode_mem, &nsetups);
  check_flag(&flag, "CVodeGetNumLinSolvSetups", 1);
  flag = CVodeGetNumErrTestFails(cvode_mem, &netf);
  check_flag(&flag, "CVodeGetNumErrTestFails", 1);
  flag = CVodeGetNumNonlinSolvIters(cvode_mem, &nni);
  check_flag(&flag, "CVodeGetNumNonlinSolvIters", 1);
  flag = CVodeGetNumNonlinSolvConvFails(cvode_mem, &ncfn);
  check_flag(&flag, "CVodeGetNumNonlinSolvConvFails", 1);

  flag = CVSpilsGetWorkSpace(cvode_mem, &lenrwLS, &leniwLS);
  check_flag(&flag, "CVSpilsGetWorkSpace", 1);
  flag = CVSpilsGetNumLinIters(cvode_mem, &nli);
  check_flag(&flag, "CVSpilsGetNumLinIters", 1);
  flag = CVSpilsGetNumPrecEvals(cvode_mem, &npe);
  check_flag(&flag, "CVSpilsGetNumPrecEvals", 1);
  flag = CVSpilsGetNumPrecSolves(cvode_mem, &nps);
  check_flag(&flag, "CVSpilsGetNumPrecSolves", 1);
  flag = CVSpilsGetNumConvFails(cvode_mem, &ncfl);
  check_flag(&flag, "CVSpilsGetNumConvFails", 1);
  flag = CVSpilsGetNumRhsEvals(cvode_mem, &nfeLS);
  check_flag(&flag, "CVSpilsGetNumRhsEvals", 1);

  printf("\nFinal Statistics.. \n\n");
  printf("lenrw   = %5ld     leniw   = %5ld\n", lenrw, leniw);
  printf("lenrwLS = %5ld     leniwLS = %5ld\n", lenrwLS, leniwLS);
  printf("nst     = %5ld\n"                  , nst);
  printf("nfe     = %5ld     nfeLS   = %5ld\n"  , nfe, nfeLS);
  printf("nni     = %5ld     nli     = %5ld\n"  , nni, nli);
  printf("nsetups = %5ld     netf    = %5ld\n"  , nsetups, netf);
  printf("npe     = %5ld     nps     = %5ld\n"  , npe, nps);
  printf("ncfn    = %5ld     ncfl    = %5ld\n\n", ncfn, ncfl);
}

/* Check function return value...
     opt == 0 means SUNDIALS function allocates memory so check if
              returned NULL pointer
     opt == 1 means SUNDIALS function returns a flag so check if
              flag >= 0
     opt == 2 means function allocates memory so check if returned
              NULL pointer */

static int check_flag(void *flagvalue, char *funcname, int opt)
{
  int *errflag;

  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
  if (opt == 0 && flagvalue == NULL) {
    fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return(1); }

  /* Check if flag < 0 */
  else if (opt == 1) {
    errflag = (int *) flagvalue;
    if (*errflag < 0) {
      fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
              funcname, *errflag);
      return(1); }}

  /* Check if function returned NULL pointer - no memory allocated */
  else if (opt == 2 && flagvalue == NULL) {
    fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return(1); }

  return(0);
}

/*
 *-------------------------------
 * Functions called by the solver
 *-------------------------------
 */

/* f routine. Compute RHS function f(t,u). */

static int f(realtype t, N_Vector u, N_Vector udot, void *user_data)
{ 
  realtype uperiodic[MX+4];
  realtype dx,lo_6,dx2,
           Hi,Him1,Hip1,Him2,Hip2,
           Him1_3,Hi_3,Hip1_3,
           Him1_6,Hi_6,Hip1_6,
           D_P_im1,D_P_i,D_P_ip1,
           Hip0p5,Him0p5,
           Hip0p5_3,Him0p5_3,
           dD_P_ip0p5,dD_P_im0p5;

  realtype *udata,*dudata;

  int lx;           
  
  UserData data;

  data = (UserData) user_data;
  udata = NV_DATA_S(u);
  dudata = NV_DATA_S(udot);

  dx = data->dx;
  dx2 = dx*dx;
  lo_6 = data->lo_6;
  
  for(lx = 0;lx<MX;lx++)
  {
    uperiodic[lx + 2] = udata[lx];
  }
  uperiodic[0] = uperiodic[MX];
  uperiodic[1] = uperiodic[MX+1];
  uperiodic[MX+2] = uperiodic[2];
  uperiodic[MX+3] = uperiodic[3];
  

  for(lx = 0;lx<MX;lx++)
  {
    Hi   =  uperiodic[lx+2];
    Him1 =  uperiodic[lx+1];
    Him2 =  uperiodic[lx];
    Hip1 =  uperiodic[lx+3];
    Hip2 =  uperiodic[lx+4];


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
           Him1pHim2,Him1pHim2_3,
           Hip2pHip1,Hip2pHip1_3,
           D_P_i,D_P_im1,D_P_ip1,
           A,B,K; //A and B are differences in disjoining pressures
  
  realtype yperiodic[MX+4];

  int jx,jy,ier;
  realtype *ydata;
//  realtype **P,**J;
  DlsMat P, J;
  long int *pivot;

  UserData data;
  
  /* Make local copies of pointers in user_data, and of pointer to u's data */
  
  data = (UserData) user_data;
  
  P = data->P;
  J = data->Jac;
  pivot = data->pivot;

  ydata = NV_DATA_S(u);

  if (jok) {
    
    /* jok = TRUE: Copy Jbd to P */
    
//    bandCopy(J,P,MX,MX-1,MX-1,2,2);    
    BandCopy(J,P,2,2);
    *jcurPtr = FALSE;
    
  }
  
  else {
    /* jok = FALSE: Generate Jbd from scratch and copy to P */
    
    /* Make local copies of problem variables, for efficiency. */
    
    dx = data->dx;
    lo_6 = data->lo_6;
    dx2 = dx*dx;
    
    
    for(jx = 0;jx<MX;jx++)
    {
      yperiodic[jx + 2] = ydata[jx];
    }
    yperiodic[0] =    yperiodic[MX];
    yperiodic[1] =    yperiodic[MX+1];
    yperiodic[MX+2] = yperiodic[2];
    yperiodic[MX+3] = yperiodic[3];


   for(jx =0;jx<MX;jx++){
           
     Hi   =  yperiodic[jx+2];
     Him1 =  yperiodic[jx+1];
     Him2 =  yperiodic[jx];
     Hip1 =  yperiodic[jx+3];
     Hip2 =  yperiodic[jx+4];
     
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

     Him1pHim2 = Him1 + Him2;
     Him1pHim2_3 = pow(Him1pHim2,3);
     
     Hip2pHip1 = Hip2 + Hip1;
     Hip2pHip1_3 = pow(Hip2pHip1,3);
      
     D_P_im1 = ((Hi -2*Him1 + Him2)/dx2) - ( (1 - lo_6/Him1_6)/(3*Him1_3)); 

     D_P_i  = ((Hip1 -2*Hi + Him1)/dx2) - ( (1 - lo_6/Hi_6)/(3*Hi_3) );
    
     D_P_ip1  = ((Hip2 -2*Hip1 + Hi)/dx2) - ( (1 - lo_6/Hip1_6)/(3*Hip1_3)) ;
     
     A = D_P_ip1 - D_P_i;   
     B = D_P_i - D_P_im1;


     K = ((3/dx2)-(1/Hi_4)+(3*lo_6/Hi_10));

     for(jy =0;jy<MX;jy++){
       
       /* dfi/dHi along diagonal */
       if(jx==jy)
//       J[jx][jy] =
       BAND_ELEM(J,jx,jy)  = 
       -((HipHip1_3*K) + (3*A*HipHip1_2) + (HipHim1_3*K) - (3*B*HipHim1_2))/(8*dx2); 
       
       /*dfi-1/dHi along upper diagonal */  
       else if((jx+1)==jy)
       BAND_ELEM(J,jx,jy)  = 
         -((HipHim1_3*-K) + (3*B*HipHim1_2) + (Him1pHim2_3/dx2))/(8*dx2);
       
        /*dfi-2/dHi along upper diagonal */
       else if((jx+2)==jy)
       BAND_ELEM(J,jx,jy)  = 
         -(Him1pHim2_3)/(8*dx2*dx2);

        /*dfi+1/dHi along the lower diagonal */
       else if((jx-1)==jy)
       BAND_ELEM(J,jx,jy)  = 
         -((-Hip2pHip1_3/dx2) - (3*A*HipHip1_2) - (HipHip1_3*K))/(8*dx2);

        /*dfi+2/dHi along the lower diagonal */
       else if((jx-2)==jy) 
       BAND_ELEM(J,jx,jy)  = 
         -(Hip2pHip1_3)/(8*dx2*dx2);

     } 

  }
  
//    bandCopy(J,P,MX,MX-1,MX-1,2,2);
    BandCopy(J,P,2,2);  
    *jcurPtr = TRUE;
    
  }
  
  /* Scale by -gamma */
  
//  bandScale(-gamma,P,MX,2,2,MX-1);
  BandScale(-gamma,P); 

  /*Add Identity matrix and do LU decompostions*/

  AddIdentity(P);
  ier = BandGBTRF(P,pivot);
  if (ier != 0) return(1);

  return(0);
}

/* Preconditioner solve routine */

static int PSolve(realtype tn, N_Vector u, N_Vector fu,
                  N_Vector r, N_Vector z,
                  realtype gamma, realtype delta,
                  int lr, void *user_data, N_Vector vtemp)
{
  DlsMat P;
  long int *pivot;
  realtype *zdata;

  UserData data;
  data = (UserData) user_data;

  /*Get P and pivot array from data*/
  P = data->P;
  pivot = data->pivot;
  zdata = NV_DATA_S(z);

  N_VScale(ONE, r, z);//Copying r to z

  /*Solve the banded system Px = r using LU factors stored in P and pivot data in pivot and return the solution to z*/
  
  BandGBTRS(P,pivot,zdata);

  return(0);
}

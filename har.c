/* ----------------------- Compiler Libraries ------------------------------- */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
/* -------------------------------------------------------------------------- */
/*---------------- Scientific Computing Libraires --------------------------- */
#include <cuda_runtime.h>
#include "magma_v2.h"
#include <cuda.h>
#include "magma_lapack.h"
/* -------------------------------------------------------------------------- */

/* ----------------------- My routines -------------------------------------- */
#include "matrix_routines/read_matrix.c"
#include "matrix_routines/allocate_matrices.c"
#include "har.h"
#include "chebyshev/chebyshev_center.c"
#include "direction_creation/random_normal_sample.c"
#include "matrix_routines_device/allocate_matrices_device.c"
#include "matrix_routines_device/multiply_matrices_device.c"
/*----------------------------------------------------------------------------*/

/*----------------------------- RM1 --------------------------------------------
------------------ Parameters for setting matrix sizes ---------------------- */
// Number of variables
static unsigned N;
// Number of Equality restrictions
static unsigned ME;
// Number of Inequalities
static unsigned MI;
// Padding for X
static unsigned Z;
// Iterations
static unsigned Iter;
// Existance of Equality Matrix
static unsigned Existance;
/* -------------------------------------------------------------------------- */

/* -------------------------- Restriction files ----------------------------- */
// For reading constraint matrices from text files
#define AE_TXT "input_matrices/A_equality.txt"
#define AI_TXT "input_matrices/A_inequality.txt"

// For reading constraint vectors form txt files
#define bE_TXT "input_matrices/b_equality.txt"
#define bI_TXT "input_matrices/b_inequality.txt"
/* -------------------------------------------------------------------------- */

/* ------------------------ Pointers to restrictions ------------------------ */
// Pointer to Matrix of Equalities
double *H_AE;
double *D_AE;
// Pointer to Matrix of Inequalities
double *H_AI;
double *D_AI;
// Pointer to Vector of Equalities
double *H_bE;
double *D_bE;
// Pointer to Vector of Inequalities
double *H_bI;
double *D_bI;

// Pointer to Porjection Matrix
double *D_PR;

// Pointer to B matrix N columns, each column is the vector BI
double *H_BI;
double *D_B; //  Will be copied to D_BS at the beggining of each loop since
            // MAGMA routines store matrices in the operation matrix C

// Pointer to D, each column is a direction of size n, square (best choice)
double *H_D;
double *D_D;

// Pointer to X, each column is a diferent walk
double *H_X;
double *D_X;

// Pointer to B_slack:= B - AI*X. No need to have AI*X (can be on the run)
double *H_BS;
double *D_BS; // Will be B at the beegining of each iteration, then it will cont
              // ain  B - AI*X

// Pointer to AID
double *H_AI_D; // To check negativity conditions
double *D_AI_D; // To multiply in device

// Pointer to LAMBDA = (B - AI*X)/(AI*D)
//double *H_LAM;
//double *D_LAM;

// Pointer to Gamma. Which mask negative and positive coefficients of LAMBDA
//int *H_GAMMA;

/* -------------------------------------------------------------------------- */

/* ------------------------ Direction Vector-Matrix ------------------------- */
// Global Rnadom Seed
double * normal_direction;
double * x_0;
/* -------------------------------------------------------------------------- */

/* ---------------------------- MAGMA queue --------------------------------- */
// Global queue for magma
magma_queue_t queue;// cudaMalloc status
magma_int_t dev;// CUBLAS functions status
/* -------------------------------------------------------------------------- */


/* *************************** init_magma *************************************/
void init_magma(){
  magma_init (); // initialize Magma
  queue = NULL ;
  dev = 0;
  magma_queue_create(dev, &queue);
}
/******************************************************************************/


/* *************************** finalize_magma *********************************/
void finalize_magma(){
  magma_queue_destroy(queue); // destroy queue
  magma_finalize (); // finalize Magma
}
/******************************************************************************/


/* ******************** allocate_matrices_host_routine ************************/
int allocate_matrices_host_har(int verbose){
  // Aux varaible for identifying the number of restrictions
  unsigned NE = 0;
  unsigned NI = 0;

  // Parse Equality vector for rows
  FILE *b_eq    = fopen(bE_TXT, "r");
  ME = count_rows(b_eq);
  b_eq = fopen(bE_TXT, "r");
  H_bE = allocate_matrices_host(b_eq, ME);
  fclose(b_eq);

  // Parse Inequality vector for rows
  FILE *b_in    = fopen(bI_TXT, "r");
  MI = count_rows(b_in);
  b_in    = fopen(bI_TXT, "r");
  H_bI = allocate_matrices_host(b_in, MI);
  fclose(b_in);

  // Parse the number of variables (columns) in AE
  FILE *a_eq    = fopen(AE_TXT, "r");
  NE       = count_columns(a_eq);
  a_eq    = fopen(AE_TXT, "r");
  H_AE = allocate_matrices_host(a_eq, NE*ME);
  fclose(a_eq);

  // Parse the number of variables (columns) in AI
  FILE *a_in    = fopen(AI_TXT, "r");
  NI = count_columns(a_in);
  a_in    = fopen(AI_TXT, "r");
  H_AI = allocate_matrices_host(a_in, NI*MI);
  fclose(a_in);

  // This condition is only tiggered if some matrix is empty
  if((NI != NE) && (NE != 0)){
    printf(" THE NUMBER OF VARIABLES IN EACH MATRIX DIFFERS. WE WILL \
    TAKE THE NUMBER OF VARIABLES (COLUMNS) IN THE INEQUALITY RESTRICTIONS \
    AS %u.\n", N);
  }else if(NE == 0){
    printf("\n NO EQUALITY RESTRICTIONS PROVIDED \n" );
    Existance = 0;
  }else if( NI == NE){
    Existance = 1;
  }
  N = NI;

  // Print
  if (verbose > 1){
    printf("Number of (rows) in the Equality Vector: %u \n", ME );
    printf("Number of (rows) in the Inequality Vector: %u \n", MI );
    printf("Number of variables (columns) in the Equality Matrix is: %u \n", NE );
    printf("Number of variables (columns) in the Inquality Matrix is: %u \n", NI );
   }

   if (verbose > 2){
      printf("\n Equality Matrix \n");
      print_matrix_debug(H_AE, ME, NE);
      printf("\n Equality Vector \n");
      print_matrix_debug(H_bE, ME, 1);
      printf("\n Inquality Matrix \n");
      print_matrix_debug(H_AI, MI, NI);
      printf("\n Inequality Vector \n");
      print_matrix_debug(H_bI, MI, 1);
   }

  return 0;

} // end allocate_matrices_host_har
/******************************************************************************/


/* ********************** generate_direction_vector ***************************/
void generate_direction_vector(unsigned vector_size, int verbose){
  for(int i = 0; i < vector_size; i++){
    normal_direction[i] = box_muller();
  }
  if (verbose > 2){
    print_matrix_debug(normal_direction, vector_size, 1);
  }
}
/******************************************************************************/

/* ********************* Create Projection Matrix *****************************/
int projection_matrix(int verbose){
  magma_int_t err ; // error handler for MAGMA library
  // Allocate AE matrix via pinned MAGMA routine

  // Constants for multiplications
  double alpha = 1.0;
  double beta = 0.0;

  // Pin A matrix in host using MAGMA routine
  H_AE = pin_matrices_host(&H_AE, ME, N);
  if(verbose > 2){
    printf("\n AE in Host \n" );
    print_matrix_debug(H_AE, ME, N);
  }

  // Allocate AE' in device
  allocate_matrices_device(H_AE, &D_AE, N, ME, queue, dev, 0);
  if(verbose > 2){
    printf("\n D_AE is Transposed in Routine \n" );
    print_matrices_in_routine(N, ME, D_AE, 1,queue);
  }

  // Obtain AA'
  double *d_AAT; // AA' device pointer
  err = magma_dmalloc (&d_AAT, ME*ME); // Allocate space for AA' in device
  matrix_multiplication_device(H_AE, H_AE, &d_AAT, N, N, ME, ME,
    1, 0, alpha, beta, queue); // Compute AA'
  if(verbose > 2){
    printf("\n  AA' in routine \n" );
    print_matrices_in_routine(ME , ME, d_AAT, 1,queue);

  }

  // Obtain (AA')^⁻1
  double *d_AAT_INV; // (AA')-1 device pointer
  d_AAT_INV = calculate_inverse_qr(d_AAT, ME, queue);
  if(verbose > 2){
    printf("\n (AA')^-1 in routine \n" );
    print_matrices_in_routine(ME, ME, d_AAT_INV, 1,queue);
  }
  magma_free(d_AAT); // We dont need this provisional matrix anymore

  // Obtain (AA')^⁻1(A)
  double * d_AAT_INV_A;
  err = magma_dmalloc (&d_AAT_INV_A, ME*N); // Allo_dev (AAT)⁻1(A)
  matrix_multiplication_device(d_AAT_INV, D_AE, &d_AAT_INV_A, ME, N, ME, ME,
  0, 1, alpha, beta, queue);
  if(verbose > 2){
    printf("\n (AA')^-1A in routine \n" );
    print_matrices_in_routine(ME, N,  d_AAT_INV_A, 1,queue);
  }
  magma_free(d_AAT_INV); // We dont need this provisional matrix anymore

  // Obtain A'(AA')^-1(A)
  double * d_AT_AAT_INV_A;
  err = magma_dmalloc (&d_AT_AAT_INV_A, N*N); // Allo_dev (AAT)⁻1(A)
  matrix_multiplication_device(D_AE, d_AAT_INV_A, &d_AT_AAT_INV_A,
  N, ME, N, ME, 0, 0, alpha, beta, queue);
  if(verbose > 2){
    printf("\n A'(AA')^-1(A) in routine\n" );
    print_matrices_in_routine(N, N,  d_AT_AAT_INV_A, 1,queue);
  }
  magma_free(d_AAT_INV_A); // We dont need this provisional matrix anymore

  // Compute I - A'(AA')^-1(A)
  err = magma_dmalloc (&D_PR, N*N);
  D_PR = allocate_identity_device(N, queue); // Fill projection Matrix as I
  if(verbose > 2){
    printf("\n I in routine \n" );
    print_matrices_in_routine(N, N,  D_PR, 1,queue);
  }
  alpha = -1.0;
  beta = 1.0;
  matrix_multiplication_device(d_AT_AAT_INV_A, D_PR, &D_PR,
  N, N, N, N, 0, 0, alpha, beta, queue);
  if(verbose > 2){
    printf("\n I - A'(AA')^-1(A) in routine \n" );
    print_matrices_in_routine(N, N,  D_PR, 1,queue);
  }
  magma_free(d_AT_AAT_INV_A);
  return 0;
}
/******************************************************************************/


/* ************************* Create B Matrix **********************************/
int create_B_matrix(int verbose){
  // Transposed in host, transposed in device->No transposed for routines
  magma_int_t err ; // error handler
  err = magma_dmalloc_pinned(&H_BI, MI*Z);
  for( int i = 0; i < Z; i++){
    for(int j = 0; j < MI; j++){
      H_BI[i*MI + j] = H_bI[j];
    }
  }
  if(verbose > 2){
    printf("\n B Transposed in Host \n" );
    print_matrix_debug(H_BI, Z, MI);
  }
  allocate_matrices_device(H_BI, &D_B, MI, Z, queue, dev, 0);
  if(verbose > 2){
    printf("\n B in Routine \n" );
    print_matrices_in_routine(MI, Z,  D_B, 1,queue);
  }
  return 0;
}
/******************************************************************************/


/* ************************** Pin D Matrix ************************************/
int pin_D_matrix(int verbose){
  magma_int_t err ; // error handler
  // Since they are iidd it does not mater if it is transposed or not when allocated
  // but then it will be transposed in host
  // Fill initial D matrix with iid observations
  err = magma_dmalloc_pinned(&H_D, N*Z);
  for( int i = 0; i < N*Z; i++){
    H_D[i] = box_muller();
  }
  allocate_matrices_device(H_D, &D_D, N, Z, queue, dev, 0);
  if(verbose > 2){
    double * h_d;
    h_d = malloc(N*Z*sizeof(double));
    magma_dgetmatrix(N, Z, D_D, N, h_d, N, queue);
    printf("\n D in Host is transposed\n" );
    print_matrix_debug_transpose(h_d, Z, N);
    free(h_d);
  }
  return 0;
}
/******************************************************************************/


/* ************************** Pin X Matrix ************************************/
int pin_X_matrix(int verbose){
  // Transposed in host, transposed in device->No transposed for routines
  magma_int_t err ; // error handler
  err = magma_dmalloc_pinned(&H_X, N*Z);
  // Fill X matrix with the initial vector in each column Z times
  for( int i = 0; i < N; i++){
    for(int j = 0; j < Z; j++){
      H_X[j*N + i] = x_0[i];
    }
  }
  if(verbose > 2){
    printf("\n X transposed in host \n" );
    print_matrix_debug(H_X, Z, N);
  }
  allocate_matrices_device(H_X, &D_X, N, Z, queue, dev, 0);
  if(verbose > 2){
    printf("\n X in routine \n" );
    print_matrices_in_routine(N, Z,  D_X, 1,queue);
  }
  return 0;
}
/******************************************************************************/

/* ************************** Pin AI Matrix ***********************************/
int init_device_AI_matrix(int verbose){
  // Transposed in host, transposed in device->No transposed for routines
  allocate_matrices_device(H_AI, &D_AI, MI, N, queue, dev, 1);
  if(verbose > 2){
    printf("\n AI in routine \n" );
    print_matrices_in_routine(MI, N,  D_AI, 0,queue);
  }
  return 0;
}
/******************************************************************************/

/* ************************** Pin AI_D Matrix *********************************/
int pin_AI_D_matrix(int verbose){
  // Transposed in host, transposed in device->No transposed for routines
  magma_int_t err ; // error handler
  err = magma_dmalloc_pinned(&H_AI_D, MI*Z);
  // Fill X matrix with the initial vector in each column Z times
  for( int i = 0; i < Z; i++){
    for(int j = 0; j < MI; j++){
      H_AI_D[i*MI + j] = -1.0;
    }
  }
  if(verbose > 2){
    printf("\n AI_D transposed in host \n" );
    print_matrix_debug(H_AI_D, Z, MI);
  }
  allocate_matrices_device(H_AI_D, &D_AI_D, MI, Z, queue, dev, 0);
  if(verbose > 2){
    printf("\n AI_D in routine \n" );
    print_matrices_in_routine(MI, Z,  D_AI_D, 1,queue);
  }
  return 0;
}
/******************************************************************************/

/* *************************** interior_point *********************************/
double * interior_point(double * x_0,int verbose){
  // Find center of the polytope using lpsolver
  x_0 = (double *)malloc(N*sizeof(double));
  //x_0 = get_initvalue(N, ME, MI, H_AE, H_bE, H_AI, H_bI);
  //print_matrix_debug(x_0, N, 1);
  // For now we will ommit chebyshev chebyshev center
  for(int n = 0; n<N; n++){
    x_0[n] = (double) 1.0/(double)N;
  }

  x_0[0] = .25;
  x_0[1] = .25;
  x_0[2] = .25;
  x_0[3] = .25;

  if (verbose >= 3){
    printf("\nInterior Point\n");
    print_matrix_debug(x_0, N, 1);
    printf("\n");
  }
  return x_0;
}// end of interior
/******************************************************************************/

/* ****************************** Fill matrix 0s ******************************/

int fill_matrix_zeros(double ** A, int m, int n){
  for(int i = 0; i < m*n; i ++){
    (*A)[i] = 0.0;
  }
  return 1;
}
/******************************************************************************/

/* ******************************* MHAR ***************************************/
void mhar(int verbose, int iters){
  /* Assumes the matrices have been already been pinned and allocated*/

  magma_int_t err ; // error handler
  err = magma_dmalloc (&(D_BS) , MI*Z); // allocate memory in device
  err = magma_dmalloc_pinned(&H_BS, MI*Z);

  double l_neg, l_pos, theta;
  int control_min =0; // Counter for first negative lambda
  int control_max =0; // Counter for first positive lambda
  double aux_coeff = 0;
  int pos = 0;

  // MHAR loop
  int iters_f = (int) ((double)iters / 10.0);
  if (iters_f == 0){
    iters_f = 1;
  }
  for( int t=0; t < iters; t++){
    if(((t % iters_f) == 0)){
      printf("\n Inter = %d of %d \n",t, iters );
    }


    // Project D
    if (Existance){
      matrix_multiplication_device(D_PR, D_D, &D_D,
      N, N, Z, N, 0, 0, 1.0, 0.0, queue);
      if(verbose > 2){
        printf("\n D Projected in routine \n" );
        print_matrices_in_routine(N, Z,  D_D, 1,queue);
      }
    }

    // Compute AI*D
    matrix_multiplication_device(D_AI, D_D, &D_AI_D,
    MI, N, Z, N, 0, 0, 1.0, 0.0, queue);
    if(verbose > 2){
      printf("\n D_AI_D in routine \n" );
      print_matrices_in_routine(MI, Z,  D_AI_D, 1,queue);
    }

    // Copy B to BS in order to do BS - AIX = BS without affecting B
    magmablas_dlacpy(MagmaFull, MI, Z, D_B, MI, D_BS, MI, queue);
    if(verbose >2){
      printf("\n BS init in routine \n" );
      print_matrices_in_routine(MI, Z,  D_BS, 1, queue);
    }

    // Compute B - AI_X=BS. This overwrites the values stored in BS
    matrix_multiplication_device(D_AI, D_X, &D_BS,
    MI, N, Z, N, 0, 0, -1.0, 1.0, queue);
    if(verbose >2){
      printf("\n BS in routine \n" );
      print_matrices_in_routine(MI, Z,  D_BS, 1, queue);
    }

    // Bring AI*D
    magma_dgetmatrix(MI, Z, D_AI_D, MI, H_AI_D, MI, queue);
    if(verbose >2){
      printf("\n AI_D in Dev \n" );
      print_matrix_debug_transpose(H_AI_D, MI, Z);
    }

    //Bring BS
    magma_dgetmatrix(MI, Z, D_BS, MI, H_BS, MI, queue);
    if(verbose >2){
      printf("\n D_BS in Dev \n" );
      print_matrix_debug_transpose(H_BS, MI, Z);
    }

    //Bring X
    magma_dgetmatrix(N, Z, D_X, N, H_X, N, queue);
    if(verbose > 2){
      printf("\n X in HOST IS Transposed \n" );
      print_matrix_debug_transpose(H_X, N, Z);
    }

    //Bring D
    magma_dgetmatrix(N, Z, D_D, N, H_D, N, queue);
    if(verbose > 2){
      printf("\n D in HOST IS Transposed \n" );
      print_matrix_debug_transpose(H_D, N, Z);
    }

    for(int pad = 0; pad < Z; pad ++){ // Loop over the padding

      control_min = 0;
      control_max = 0;

      for(int m_i = 0; m_i < MI; m_i ++){ // Loop over the possible restrictions
        pos = pad*MI + m_i;
        aux_coeff =  H_BS[pos]/ H_AI_D[pos]; // Fraction (B-AX) / AD
        if(H_AI_D[pos] < 0.0){ // The denominator is negative
          if(control_min == 0){ l_neg = aux_coeff, control_min=1;}
          else if(aux_coeff > l_neg){l_neg = aux_coeff;}
        } // ends negative lambda if
        if(H_AI_D[pos] > 0.0){ // The denominator is positive
          if(control_max == 0){ l_pos = aux_coeff, control_max=1;}
          else if(aux_coeff < l_pos){l_pos = aux_coeff;}
        } // ends positive lambda if
      }

      // Find theta
      theta = ldexp(pcg64_random(), -64);
      while(theta == 0.0){ // The generator may generate 0.0, so we iterate again
        theta = ldexp(pcg64_random(), -64);
      }
      theta = theta*l_pos + (1.0-theta)*l_neg; // Convex combination

      // Make the linear combination
      for(int pos_x = 0; pos_x < N; pos_x++){
        H_X[pos_x + pad*Z] = H_X[pos_x + pad*Z] + theta * H_D[pos_x + pad*Z];
      }
    } // End New Point Loop

    // Send X to device
    allocate_matrices_device(H_X, &D_X, N, Z, queue, dev, 0);
    // Generate new D
    for( int i = 0; i < N*Z; i++){
      H_D[i] = box_muller();
    }
    allocate_matrices_device(H_D, &D_D, N, Z, queue, dev, 0); // Send D to Device
    if(verbose >2){
      printf("\n D_BS in routine \n" );
      print_matrices_in_routine(MI, Z,  D_BS, 1, queue);
    }
  }
}
/******************************************************************************/


/* ********************** free allocated host matrices ************************/
int free_host_matrices_har(){
  magma_free_pinned(H_AE);
  magma_free_pinned(H_AI_D);
  magma_free_pinned(H_BI);
  magma_free_pinned(H_D);
  magma_free_pinned(H_X);
  magma_free_pinned(H_BS);

  free(H_AI);
  free(H_bE);
  free(H_bI);
  return 0;
}// End free_host_matrices_har
/******************************************************************************/


/************************ free allocated device matrices **********************/
int free_device_matrices_har(){
  magma_free (D_AE);
  magma_free (D_AI);
  magma_free (D_AI_D);
  if(Existance){
    magma_free (D_PR);
  }
  magma_free (D_B);
  magma_free (D_D);
  magma_free (D_X);
  magma_free (D_BS);

  return 0;
}// End free_device_matrices_har
/******************************************************************************/


/* ******************************* Main ***************************************/
int main(){
  /* verbose
  * 0 nothing
  * 1 Only time
  * 2 Prints Dimensions
  * 3 Prints Matrices
  */
  int verbose = 3;
  double time_spent;
  clock_t begin;
  clock_t end ;

  // Iterations
  // Allocate matrices in host
  begin = clock();
  allocate_matrices_host_har(verbose);
  end = clock();
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  if (verbose > 0){
    printf("\n ---- Time allocate_matrices_host_har: %lf\n", time_spent);
  }
  // End matrices in host

  // Find Interior Pointprintf ( "\n after geqrf : info_gpu = %d \n " , info_gpu );

  begin = clock();
  x_0 = interior_point(x_0, verbose);
  end = clock();
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  if (verbose > 0){
    printf("\n ---- Time for Chebyshev Center: %lf\n", time_spent);
  }
  // End Interiror Point

  // Init random seed
  begin = clock();
  init_random_seeds();
  end = clock();
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  if (verbose > 0){
    printf("\n ---- Init Random Seeds: %lf\n", time_spent);
  }
  // End init random seed

  // Create MAGMA Context
  init_magma();


  // Calculate Projection Matrix
  if (Existance){
    begin = clock();
    projection_matrix(verbose);
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    if (verbose > 0){
      printf("\n ---- Projection Matrix: %lf\n", time_spent);
    } // End calculate projection Matrix
  }


  // Initialize padding
  Z = 1;
  Iter = 5;

  // Create BI matrix
  begin = clock();
  create_B_matrix(verbose);
  end = clock();
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  if (verbose > 0){
    printf("\n ---- Create B: %lf\n", time_spent);
  }
  // End create BI matrix

  // Initialize D
  begin = clock();
  pin_D_matrix(verbose);
  end = clock();
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  if (verbose > 0){
    printf("\n ---- Init and Pin D Host & Device: %lf\n", time_spent);
  }
  // End Initialize D

  // Initialize X
  begin = clock();
  pin_X_matrix(verbose);
  end = clock();
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  if (verbose > 0){
    printf("\n ---- Init and Pin X Host & Device: %lf\n", time_spent);
  }
  // End Initialize X

  // Initialize AI
  begin = clock();
  init_device_AI_matrix(verbose);
  end = clock();
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  if (verbose > 0){
    printf("\n ---- Init Pin AI Device: %lf\n", time_spent);
  }
  // End Initialize AI

  // Initialize AI_D
  begin = clock();
  pin_AI_D_matrix(verbose);
  end = clock();
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  if (verbose > 0){
    printf("\n ---- Init Pin AI_D: %lf\n", time_spent);
  }
  // End Initialize AI_D


  // MHAR
  begin = clock();
  mhar(verbose, Iter);
  end = clock();
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  if (verbose > 0){
    printf("\n ---- MHAR: %lf\n", time_spent);
  }
  // MHAR


  // Free allocated matrices in the host
  begin = clock();
  free_host_matrices_har();
  end = clock();
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  if (verbose > 0){
    printf("\n ---- free_host_matrices_har: %lf\n", time_spent);
  }
  // End Free allocated matrices in the host

  // Free allocated matrices in the host
  begin = clock();
  free_device_matrices_har();
  end = clock();
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  if (verbose > 0){
    printf("\n ---- free_device_matrices_har: %lf\n", time_spent);
  }
  // End Free allocated matrices in the host

  // End MAGMA context
  //magma_queue_destroy(queue);
  finalize_magma();
  return 0;
} // End Main
/******************************************************************************/

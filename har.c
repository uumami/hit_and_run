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
/* ------------------------------------------------------------------------- */

/* *************************** init_magma *********************************** */
void init_magma(){
  magma_init (); // initialize Magma
  queue = NULL ;
  dev = 0;
  magma_queue_create(dev, &queue);
}
/******************************************************************************/

/* *************************** finalize_magma ******************************* */
void finalize_magma(){
  magma_queue_destroy(queue); // destroy queue
  magma_finalize (); // finalize Magma
}
/******************************************************************************/

/* ******************** allocate_matrices_host_routine *********************  */
int allocate_matrices_host_har(int verbose){
  // Aux varaible for identifying the number of restrictions
  unsigned NE = 0;
  unsigned NI = 0;

  // Parse Equality vector for rows
  FILE *b_eq    = fopen(bE_TXT, "r");
  ME = count_rows(b_eq);
  b_eq    = fopen(bE_TXT, "r");
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
  if(NI != NE){
    printf(" THE NUMBER OF VARIABLES IN EACH MATRIX DIFFERS. WE WILL \
    TAKE THE NUMBER OF VARIABLES (COLUMNS) IN THE EQUALITY RESTRICTIONS \
    AS %u.\n", N);
  }
  N = NE;

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

/* ********************* Create Projection Matrix *****************************/
void projection_matrix(int verbose){
  // Allocate AE matrix via pinned MAGMA routine
  H_AE = pin_matrices_host(&H_AE, ME, N, queue, dev);
  if(verbose >2){
    printf("\n Matrix Equality allocated via MAGMA pinned routine \n" );
    print_matrix_debug(H_AE, ME, ME);
  }
}
/******************************************************************************/

/***************************** interior_point *********************************/
double * interior_point(double * x_0,int verbose){
  // Find center of the polytope using lpsolver
  x_0 = (double *)malloc(N*sizeof(double));
  //x_0 = get_initvalue(N, ME, MI, H_AE, H_bE, H_AI, H_bI);
  //print_matrix_debug(x_0, N, 1);
  // For now we will ommit chebyshev chebyshev center
  for(int n = 0; n<N; n++){
    x_0[n] = (double) 1.0/(double)N;
  }

  if (verbose >= 3){
    printf("\nInterior Point\n");
    print_matrix_debug(x_0, N, 1);
    printf("\n");
  }
  return x_0;
}// end of interior
/******************************************************************************/


/************************ generate_direction_vector ***************************/
void generate_direction_vector(unsigned vector_size, int verbose){
  for(int i = 0; i < vector_size; i++){
    normal_direction[i] = box_muller();
  }
  if (verbose > 2){
    print_matrix_debug(normal_direction, vector_size, 1);
  }
}
/******************************************************************************/

/************************ free allocated host matrices *********************  */
int free_host_matrices_har(){
  //free(H_AE);
  magma_free_pinned(H_AE);
  free(H_AI);
  free(H_bE);
  free(H_bI);
  return 0;
}// End free_host_matrices_har
/******************************************************************************/

/************************ free allocated device matrices *********************  */
int free_device_matrices_har(){

  return 0;
}// End free_device_matrices_har
/******************************************************************************/


/* ******************************* Main ************************************* */
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

  normal_direction = (double *)malloc(sizeof(double)*N);
  begin = clock();
  generate_direction_vector(N, verbose);
  end = clock();
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  if (verbose > 0){
    printf("\n ----Generate Normal Direction: %lf\n", time_spent);
  }

  // Create MAGMA Context
  init_magma();

  // Calculate Projection Matrix
  begin = clock();
  projection_matrix(verbose);
  end = clock();
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  if (verbose > 0){
    printf("\n ---- Projection Matrix: %lf\n", time_spent);
  } // End calculate projection Matrix

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
  finalize_magma();
  return 0;
} // End Main

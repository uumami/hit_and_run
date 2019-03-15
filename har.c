/* ----------------------- Compiler Libraries ------------------------------- */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
/* -------------------------------------------------------------------------- */
/*---------------- Scientific Computing Libraires --------------------------- */
#include <cuda_runtime.h>
#include "cublas_v2.h"
/* -------------------------------------------------------------------------- */

/* ----------------------- My routines -------------------------------------- */
#include "matrix_routines/read_matrix.c"
#include "matrix_routines/allocate_matrices.c"
#include "har.h"
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
static double *H_AE;
static double *D_AE;
// Pointer to Matrix of Inequalities
static double *H_AI;
static double *D_AI;
// Pointer to Vector of Equalities
static double *H_bE;
static double *D_bE;
// Pointer to Vector of Inequalities
static double *H_bI;
static double *D_bI;
/* -------------------------------------------------------------------------- */

/* ******************** allocate_matrices_host_routine *********************  */
int allocate_matrices_host_har(int verbose){
  // Aux varaible for identifying the number of restrictions
  unsigned NE = 0;
  unsigned NI = 0;

  // Parse Equality vector for rows
  FILE *b_eq    = fopen(bE_TXT, "r");
  ME = count_rows(b_eq);
  allocate_matrices_host(H_bI, b_eq, ME);
  fclose(b_eq);

  // Parse Inequality vector for rows
  FILE *b_in    = fopen(bI_TXT, "r");
  MI = count_rows(b_in);
  allocate_matrices_host(H_bI, b_eq, MI);
  fclose(b_in);

  // Parse the number of variables (columns) in AE
  FILE *a_eq    = fopen(AE_TXT, "r");
  NE       = count_columns(a_eq);
  allocate_matrices_host(H_bI, b_eq, NE*ME);
  fclose(a_eq);

  // Parse the number of variables (columns) in AI
  FILE *a_in    = fopen(AI_TXT, "r");
  NI = count_columns(a_in);
  allocate_matrices_host(H_bI, b_eq, NI*MI);
  fclose(a_in);

  // This condition is only tiggered if some matrix is empty
  if(NI != NE){
  printf(" THE NUMBER OF VARIABLES IN EACH MATRIX DIFFERS. WE WILL \
  TAKE THE NUMBER OF VARIABLES (COLUMNS) IN THE EQUALITY RESTRICTIONS \
  AS %u.\n", N);
  }
  N = NE;

  // Print
  if (verbose){
    printf("Number of (rows) in the Equality Vector: %u \n", ME );
    printf("Number of (rows) in the Inequality Vector: %u \n", MI );
    printf("Number of variables (columns) in the Equality Matrix is: %u \n", NE );
    printf("Number of variables (columns) in the Inquality Matrix is: %u \n", NI );
   }

  return 0;

} // end allocate_matrices_host_har

/************************ free allocated host matrices *********************  */
int free_host_matrices_har(){
  free(H_AE);
  free(H_AI);
  free(H_bE);
  free(H_bI);
  return 0;
}// End free_host_matrices_har

/* ******************************* Main ************************************* */
int main(){
  int verbose = 1;

  // Allocate matrices in host
  allocate_matrices_host_har(verbose);

  // Free allocated matrices in the host
  free_host_matrices_har();

  return 0;
} // End Main

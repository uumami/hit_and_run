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
int allocate_matrices_host_routine(){
  // Aux varaible for identifying the number of restrictions
  unsigned NE = 0;
  unsigned NI = 0;

  // Parse Equality vector for rows
  FILE *b_eq    = fopen(bE_TXT, "r");
  ME = count_rows(b_eq);
  fclose(b_eq);
  printf("Number of (rows) in the Equality Vector: %u \n", ME );

  // Parse Inequality vector for rows
  FILE *b_in    = fopen(bI_TXT, "r");
  MI = count_rows(b_in);
  fclose(b_in);
  printf("Number of (rows) in the Inequality Vector: %u \n", MI );

  // Parse the number of variables (columns) in AE
  FILE *a_eq    = fopen(AE_TXT, "r");
  NE       = count_columns(a_eq);
  fclose(a_eq);

  // Parse the number of variables (columns) in AI
  FILE *a_in    = fopen(AI_TXT, "r");
  NI = count_columns(a_in);
  fclose(a_in);
  // Print
  printf("Number of variables (columns) in the Equality Matrix is: %u \n", NE );
  printf("Number of variables (columns) in the Inquality Matrix is:\
  %u \n", NI );
  N = NE;
    // This condition is only tiggered if some matrix is empty
  if(NI != NE){
  printf(" THE NUMBER OF VARIABLES IN EACH MATRIX DIFFERS. WE WILL \
  TAKE THE NUMBER OF VARIABLES (COLUMNS) IN THE EQUALITY RESTRICTIONS \
  AS %u.\n", N);
  }

  printf("Number of Variables: %u \n", N );
  return 0;
}
/*--------------------------- Main ------------------------------------------- */
int main(){
  // Allocate matrices in host
  allocate_matrices_host_routine();
  return 0;
} // End Main

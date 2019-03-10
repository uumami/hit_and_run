/* ----------------------- Compiler Libraries ------------------------------- */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
/* -------------------------------------------------------------------------- */
/* ----------------------- Sc Comp Libraries ------------------------------- */

#include <cuda_runtime.h>
#include "cublas_v2.h"
/* -------------------------------------------------------------------------- */


/*----------------------------- RM1 --------------------------------------------
------------------ Parameters for setting matrix sizes ---------------------- */
// Number of variables
static unsigned N;
// Number of Equality restrictions
static unsigned ME;
// Number of Inequalities
static unsigned MI;
/* -------------------------------------------------------------------------- */


/*----------------------------- RM2 --------------------------------------------
/* ------------------------ Pointers to restrictions ------------------------ */
// Pointer to Matrix of Equalities
static double *P_AE;
// Pointer to Matrix of Inequalities
static double *P_AI;
// Pointer to Vector of Equalities
static double *P_bE;
// Pointer to Vector of Inequalities
static double *P_bI;
/* -------------------------------------------------------------------------- */


/* -------------------------- Restriction files ----------------------------- */
// For reading constraint matrices from text files
#define AE_TXT "input_matrices/A_equality.txt"
#define AI_TXT "input_matrices/A_inequality.txt"

// For reading constraint vectors form txt files
#define bE_TXT "input_matrices/b_equality.txt"
#define bI_TXT "input_matrices/b_inequality.txt"
/* -------------------------------------------------------------------------- */

/* ----------------------- Prototype functions ------------------------------ */

/***************************  count_rows *************************************
  * This functions counts how many lines (rows) a txt file has.
    It will be used by the function count_restrictions.
  * Requirements: None
  * Inputs:
      + The separtor between lines (rows) must be different from the varaible
        (column) separator. We recomend using the new line character.
      + Pointer to the txt File.
  * Output:
      + Returns and unsigned integer with the number of lines (rows)
      in the txt file.
*******************************************************************************/
unsigned count_rows(FILE *file);



/***************************  count_columns *************************************
  * This functions counts how many numbers per line (column) a txt file has.
    It will be used by the function count_restrictions.
  * Requirements: None
  * Inputs:
      + The separtor between values (colums) must be different from the line
        (rows) separator. We recomend using comas, spaces or pipes.
      + Pointer to the txt File.
  * Output:
      + Returns and unsigned integer with the number of variables (columns)
      in the txt file.
*******************************************************************************/
unsigned count_columns(FILE *file);


/***************************  count_restrictions ******************************
*  This functions opens the txt file and counts how many restrictions (Equality
*  and Inequalities) the txt files have and saves them in the varaibles
*  N, ME and MI.
*  Requiremntes: Functions count_lines and count variables defined above.
*  Inputs:
      + None
*  Output:
      + None, the varaibles are saved via pointers. See section RM1
      (at the beginning of this header file).
*******************************************************************************/
int parse_restrictions();


/***************************  allocate_matrices ********************************
  * This function allocates and saves the matrices in memory.
  * Requirements: count_restrictions already used in order to know N, ME, & MI
  * Inputs:
      + None
  * Output:
      + None, the matrices are allocated via the pointers defined in section RM2
      (at the beginning of this header file).
*******************************************************************************/
void read_matrices();

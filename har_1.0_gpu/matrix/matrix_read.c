/*--------- Standar Libraries-------------*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
/*-------------------------------------- */


/*--------- Algebra libraries--------- */
#include "mkl.h"
#include "mkl_service.h"
#include "i_malloc.h"
/*-------------------------------------- */


/*----- Parameters for setting matrix sizes ------- */
// Number of variables
static unsigned N;
// Number of Equalities restrictions
static unsigned ME;
// Number of Inequalities
static unsigned MI;
/* ------------------------------------------------ */



/*---------- Pointers to restrictions --------------*/

// Matrix of Equalities
static double *E;

// Matrix of Inequalities
static double *W;

// Vector of Equalities
static double *BE;

// Vector of Inequalities
static double *BW;

/* ------------------------------------------------ */



// For reading constraint matrices from text files
#define A_EQ "restrictions/A_eq.txt"
#define A_INEQ "restrictions/A_in.txt"

// For reading constraint vectors form txt files
#define b_EQ "restrictions/b_eq.txt"
#define b_INEQ "restrictions/b_in.txt"



/* ------------------  Begin Count  variables and restrictions -------------------- */
/* -------------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------------- */

void count ()
{
  // Aux varaible for identifying the number of restrictions
  int aux_var = 0;

  // Pointers to txt files
  FILE *a_eq    = fopen(A_EQ, "r");
  FILE *a_in    = fopen(A_INEQ, "r");
  FILE *b_eq    = fopen(b_EQ, "r");
  FILE *b_in    = fopen(b_INEQ, "r");

  ME = count_lines(b_eq);
  fclose(b_eq);

  MI = count_lines(b_in);
  fclose(b_in);

  printf(" NUMBER OF EQUALITIES   : %u \n", ME );
  printf(" NUMBER OF INEQUALITIES : %u \n", MI );

  N       = count_vars(a_eq);
  fclose(a_eq);

  aux_var = count_vars(a_in);
  fclose(a_in);


  // This condition is only tiggered if some matrix is empty
  if(N < aux_var){N = aux_var;}

  printf(" NUMBER OF VARIABLES    : %u \n", N );



}
/* ----------------------------------------------------------------------------*/


/*
  + Funtion for couting the number of restrictions (lines)
  + Receives a pointer to the txt file
  + Returns as an unsigned int the number of restricion (lines)
*/
int count_lines (FILE *file)
{
  int ch    =   0;
  unsigned lines =   0;

  int aux_line   =  1;
  // Debug for inexistent restrictions

  if(file == NULL)
  {
      printf("\n The program can not find the files containing the restrictions \n");
  }

  while ((ch = fgetc(file)) != EOF)
  {
    if (ch == '\n')         {aux_line = 1;}

    // 48 and 57 represents the ascii code for 0 and 9 respectibly
    if (ch <= 57 && ch >=48 && aux_line)  {lines++, aux_line=0;}
  }
  return lines;
}
/* ----------------------------------------------------------------------------*/


/*
  + Funtion for couting the number of restrictions (columns)
  + Receives a pointer to the txt file
  + Returns as an unsigned int the number of restricion (columns)
*/
int count_vars(FILE *file)
{
  int ch    =   0;
  unsigned vars  =   0;

  int aux_space  =  1;
  // Debug for inexistent restrictions

  if(file == NULL)
  {
      printf("\n The program can not find the files containing the restrictions \n");
  }

  while ((ch = fgetc(file)) != EOF && ch != '\n')
  {
    if (ch == ' ')         {aux_space = 1;}

    // 48 and 57 represents the ascii code for 0 and 9 respectibly
    if (ch <= 57 && ch >=48 && aux_space)  {vars++, aux_space=0;}
  }
  return vars;
}
/* ----------------------------------------------------------------------------*/


/* ----------------------- End Count variables and restrictions -------------------- */
/* --------------------------------------------------------------------------------- */
/* --------------------------------------------------------------------------------- */



/*-----------------------Begin Reading Matrices -------------------------------------- */
/* --------------------------------------------------------------------------------- */
/* --------------------------------------------------------------------------------- */

void read_matrices()
{
  // Pointers to txt files
  FILE *a_eq    = fopen(A_EQ, "r");
  FILE *a_in    = fopen(A_INEQ, "r");
  FILE *b_eq    = fopen(b_EQ, "r");
  FILE *b_in    = fopen(b_INEQ, "r");

  // Allocates memory for rows

  //  E   = (double *)malloc(sizeof(double) * ME * N);
  E   = (double *)mkl_malloc(sizeof(double) * ME * N,64);
  BE  = (double *)malloc(sizeof(double) * ME);

  //  W   = (double *)malloc(sizeof(double) * MI * N);
  W   = (double *)mkl_malloc(sizeof(double) * MI * N,64);
  BW  = (double *)malloc(sizeof(double) * MI);

  // Checks if memory was allocated succesfully

  if (E == NULL || W == NULL)
  {
    printf( "\n ERROR: Can't allocate memory for matrices with mkl_malloc . Aborting... \n\n");
    mkl_free(E);
    mkl_free(W);

    // mkl_free(B);

  }

  if (BE == NULL || BW == NULL)
  {
    printf( "\n ERROR: Can't allocate memory for vectors withmalloc . Aborting... \n\n");
    free(BE);
    free(BW);

    // mkl_free(B);

  }

  // Reads equality Matrices
  for (int m = 0; m < ME; m++)
  {

    fscanf( b_eq, "%lf", &(BE[m]) );

    for (int n = 0; n < N; n++)
    {
      fscanf( a_eq, "%lf", &(E[m*N + n]) );
    }

  }
  fclose(a_eq);
  fclose(b_eq);

  // Reads Inequality Matrices
  for (int m = 0; m < MI; m++)
  {

    fscanf( b_in, "%lf", &(BW[m]) );

    for (int n = 0; n < N; n++)
    {
      fscanf( a_in, "%lf", &(W[m*N + n]) );
    }

  }

  fclose(a_in);
  fclose(b_in);


}

void free_restrictions()
{
  mkl_free(E);
  mkl_free(W);
  free(BE);
  free(BW);
}



/*----------------------- End Reading Matrices -------------------------------------- */
/* --------------------------------------------------------------------------------- */
/* --------------------------------------------------------------------------------- */

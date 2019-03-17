/* ----------------------- Compiler Libraries ------------------------------- */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
/* -------------------------------------------------------------------------- */
// header
#include "allocate_matrices.h"

int allocate_matrices_host(double *P, FILE *file, unsigned size){

  if (file == NULL){
    printf("The pointer to the file is invalid \n" );
  }

  P = (double *)malloc(sizeof(double) * size);
  if (P == NULL){
    printf("The matrix pointer could not be genrated \n" );
  }

  // Reads equality Matrices
  for (int j = 0; j < size; j++){
    fscanf( file, "%lf", &(P[j]) );
  }

  return 0;

}

print_matrix_debug(double *){

}

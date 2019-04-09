/* ----------------------- Compiler Libraries ------------------------------- */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
/* ----------------------- CUDA LIBRARIES ----------------------------------- */
#include <cuda_runtime.h>
#include "cublas_v2.h"
/* ------------------------------ Header ------------------------------------ */
#include "allocate_matrices_device.h"

int allocate_matrices_device(double * h_matrix, double **d_matrix,
unsigned m, unsigned n, cublasHandle_t handle, cudaError_t cudaStat,
cublasStatus_t stat ){
  // Recall: m->row, n->column

  // Copy matrix to device
  cudaStat = cudaMalloc ((void **)&(*d_matrix) , m*n*sizeof(*h_matrix ));
  // cp h_matrix - > d_matrix
  stat = cublasSetMatrix (m, n, sizeof(*h_matrix), h_matrix, m, *d_matrix, m);
  return  1;
}

/* ----------------------- Compiler Libraries ------------------------------- */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
/* ----------------------- CUDA LIBRARIES ----------------------------------- */
#include <cuda_runtime.h >
#include "cublas_v2.h"
/* ------------------------------ Header ------------------------------------ */
#include "allocate_matrices_device.h"

void allocate_matrices_device(double * h_matrix, double * d_matrix,
unsigned m, unsigned n, cublasHandle_t handle){
  // Recall: m->row, n->column
  // cudaMalloc status
  cudaError_t cudaStat ;
  // CUBLAS functions status
  cublasStatus_t stat ;

  // Copy matrix to device
  cudaStat = cudaMalloc ((void **)&d_matrix , m*n*sizeof(*h_matrix ));
  stat = cublasCreate (&handle);
  // cp h_matrix - > d_matrix
  stat = cublasSetMatrix (m, n, sizeof(*a), a, m, d_a, m);
  return 1;
}

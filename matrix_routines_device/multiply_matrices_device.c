/* ----------------------- Compiler Libraries ------------------------------- */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
/* ----------------------- CUDA LIBRARIES ----------------------------------- */
#include <cuda_runtime.h>
#include "cublas_v2.h"
/* ------------------------------ Header ------------------------------------ */
#include "multiply_matrices_device.c"

void multiply_matrices_device(double alpha, double beta, double *d_a,
  double *d_b, cublasHandle_t handle, unsigned m, unsigned n, unsigned k){

  }

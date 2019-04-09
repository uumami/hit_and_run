/* NOTES
  THis code is based on mygpu.pdf
*/


/* ----------------------- Compiler Libraries ------------------------------- */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
/* ----------------------- CUDA LIBRARIES ----------------------------------- */
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cusolverDn.h>
#include <cblas.h>
# define BILLION 1000000000L;
/* ------------------------------ Header ------------------------------------ */

void solver_qr(double *A, double *B, unsigned m){
  // Context Handler
  cusolverDnHandle_t cusolverH; // cusolver handle
  cublasHandle_t cublasH;// cublas handle
  cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
  cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

  // variables for error checking in cudaMalloc
  cudaError_t cudaStat1 = cudaSuccess;
  cudaError_t cudaStat2 = cudaSuccess;
  cudaError_t cudaStat3 = cudaSuccess;
  cudaError_t cudaStat4 = cudaSuccess;

  // Solves X in AX=B
  // Recall m-> NUmber of rows in H
  const int lda = m ; // leading dimension of A
  const int ldb = m ; // leading dimension of B
  const int nrhs = 1; // number of right hand sides

  // Init scalars
  double al=1.0, bet=0.0;
  int incx =1 , incy =1;
  cblas_dgemv (CblasColMajor, CblasNoTrans, m, m, al, A, m, B1, incx, bet, B,
    incy); // B = A * B1

  // Destrpy Cublas context
  cublasDestroy (cublasH);
  cusolverDnDestroy (cusolverH);

}

void calculate_inverse_qr(double *A, unsigned m){

  // Create Identity Matrix for the inverse
  double *I;
  I = malloc(m*m*sizeof(double));
  for(int i=0; i<m; i++){
    for(int j=0: j<m; j++){
      if(i==j){I[i*m + j] = 1.0;}
      else{ I[i*m + j] = 0.0;}
    }
  }
  // Call solver_qr routine with identty matrix as RHS
  solver_qr(A, I, m);

  free(I);
}

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
/* ------------------------------ Header ------------------------------------ */
void solver_qr(double *A, double *B, unsigned m){
  // AX = B
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
  int incx =1, incy =1;

  // Init auxiliary matrices
  double *B1;
  double *X;
  B1 = (double *) malloc(ldb*nrhs*sizeof(double));
  X = (double *) malloc(ldb*nrhs*sizeof(double));
  // Find B1
  cblas_dgemv (CblasColMajor, CblasNoTrans, m, m, al, A, m, B1, incx, bet, B,
  incy); // B = A * B1

  // declare arrays on the device
  double * d_A, * d_B, * d_tau, * d_work;
  int * devInfo ;
  // device version of info
  int lwork = 0;
  // workspace size
  int info_gpu = 0;
  // device info copied to host
  const double one = 1;
  // create cusolver and cublas handles
  cusolver_status = cusolverDnCreate(&cusolverH);
  cublas_status = cublasCreate (&cublasH);
  // prepare memory on the device
  cudaStat1 = cudaMalloc ((void **)&d_A, sizeof(double)*lda*m);
  cudaStat2 = cudaMalloc ((void **)&d_tau, sizeof(double)*m);
  cudaStat3 = cudaMalloc ((void **)&d_B, sizeof(double)*ldb*nrhs);
  cudaStat4 = cudaMalloc ((void **)&devInfo, sizeof(int));

  cudaStat1 = cudaMemcpy (d_A, A, sizeof(double)*lda*m,
  cudaMemcpyHostToDevice);//A - > d_A
  cudaStat2 = cudaMemcpy ( d_B ,B , sizeof ( double )* ldb * nrhs ,
  cudaMemcpyHostToDevice); // B - > d_B

// compute buffer size for geqrf and prepare worksp . on device
cusolver_status = cusolverDnDgeqrf_bufferSize(cusolverH, m, m, d_A, lda, &lwork);
cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
cusolver_status = cusolverDnDgeqrf(cusolverH, m, m, d_A, lda, d_tau, d_work,
  lwork, devInfo);
cudaStat1 = cudaDeviceSynchronize();
// devInfo -> info_gpu
cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
// check error code of geqrf function
if(info_gpu){
  printf ( "\n after geqrf : info_gpu = %d \n " , info_gpu );
}
// Destroy Cublas context
free(B1);
free(X);
cublasDestroy(cublasH);
cusolverDnDestroy(cusolverH);

}

void calculate_inverse_qr(double *A, unsigned m){
  // Create Identity Matrix for the inverse
  double * ID;
  m = (int) m;
  ID = (double *) malloc(m*m*sizeof(double));
  for(int i=0; i<m; i++){
    for(int j=0; j<m; j++){
      if(i==j){
        ID[i*m + j] = 1.0;
      }
      else{
        ID[i*m + j] = 0.0;
      }
    }
  }
  // Call solver_qr routine with identty matrix as RHS
  solver_qr(A, ID, m);
  free(ID);
}

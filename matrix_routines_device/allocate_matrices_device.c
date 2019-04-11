/* ----------------------- Compiler Libraries ------------------------------- */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
/* ----------------------- CUDA LIBRARIES ----------------------------------- */
#include <cuda_runtime.h>
#include "magma_v2.h"
#include <cuda.h>
#include "magma_lapack.h"
/* ------------------------------ Header ------------------------------------ */
#include "allocate_matrices_device.h"

int allocate_matrices_device(double *h_matrix, double **d_matrix,
unsigned m, unsigned n, magma_queue_t queue, magma_int_t dev){

  // Recall: m->row, n->column
  magma_int_t m_m = m;
  magma_int_t m_n = n;

  magma_int_t err ; // error handler
  // Copy matrix to device
  //cudaStat = cudaMalloc ((void **)&(*d_matrix) , m*n*sizeof(*h_matrix ));
  // cp h_matrix - > d_matrix
  //stat = cublasSetMatrix (m, n, sizeof(*h_matrix), h_matrix, m, *d_matrix, m);
  return  1;
}


double * pin_matrices_host(double **h_matrix, unsigned m, unsigned n,
  magma_queue_t queue, magma_int_t dev){

  // Recall: m->row, n->column
  magma_int_t m_m =  m;
  magma_int_t m_n =  n;

  magma_int_t err ; // error handler

  double *pinned_matrix;
  // Pin Matrix to device
  err = magma_dmalloc_pinned(&pinned_matrix, m_m*m_n);
  for( int i=0; i < m*n; i++)
  {
    printf("\n Check two \n");
    pinned_matrix[i] = (*h_matrix)[i];
  }
  free(*h_matrix);
  return  pinned_matrix;
}

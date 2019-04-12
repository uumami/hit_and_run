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

  err = magma_dmalloc (&(*d_matrix) , m_m*m_n ); // allocate memory in device
  // Remember the matrix are col-major, then we allocate the transpose
  magma_dsetmatrix (m_n, m_m, h_matrix, m_n, *d_matrix, m_n, queue);
  return  0;
}

int allocate_matrices_device_same_dim(double *h_matrix, double **d_matrix,
unsigned m, unsigned n, magma_queue_t queue, magma_int_t dev){

  // Recall: m->row, n->column
  magma_int_t m_m = m;
  magma_int_t m_n = n;

  magma_int_t err ; // error handler

  err = magma_dmalloc (&(*d_matrix) , m_m*m_n ); // allocate memory in device
  // Remember the matrix are col-major, then we allocate the transpose
  magma_dsetmatrix_transpose(m_n, m_m, h_matrix, m_n, *d_matrix, m_n, queue);
  return  0;
}

double * pin_matrices_host(double **h_matrix, unsigned m, unsigned n){
  // Recall: m->row, n->column
  magma_int_t m_m =  m;
  magma_int_t m_n =  n;

  magma_int_t err ; // error handler

  double *pinned_matrix;// Pin Matrix to device

  err = magma_dmalloc_pinned(&pinned_matrix, m_m*m_n);
  for( int i=0; i < m*n; i++)
  {
    pinned_matrix[i] = (*h_matrix)[i];
  }
  free(*h_matrix);
  return  pinned_matrix;
}

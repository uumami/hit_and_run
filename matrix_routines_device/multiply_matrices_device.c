/* NOTES
  THis code is based on mygpu.pdf
*/
/* ----------------------- Compiler Libraries ------------------------------- */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
/* ----------------------- CUDA LIBRARIES ----------------------------------- */
#include <cuda_runtime.h>
/* ------------------------------ Header ------------------------------------ */

void matrix_multiplication_device(double *d_a, double *d_b, double **d_c,
  unsigned row_a, unsigned row_b, unsigned col_b, unsigned col_a, int trans_a,
  int trans_b, double al, double bet, magma_queue_t queue){
  // In put the matrix dimensions as they live in the device not the dim after
  // applying op(Matrix)
  // a->m*k_a(lda m), b->k*n (ldb k)
  // Magma dimension types (it takes into account the matrices are stored in
  // column major fashion or transposed)
  magma_int_t m = row_a;
  magma_int_t k = row_b;
  magma_int_t n  = col_b;
  magma_int_t k_a  = col_a;
  magma_int_t err ; // error handler

  // Create constants
  double alpha = MAGMA_S_MAKE ( al , 0.0 );
  double beta = MAGMA_S_MAKE ( bet , 0.0 );

  if(trans_a == 0 && trans_b==0){ // a->m*k_a, b->k*n, c->m*n
    //err = magma_dmalloc (d_c, m*n);
    magma_dgemm(MagmaNoTrans, MagmaNoTrans, m, n, k, alpha, d_a, m, d_b, k,
      beta, *d_c, m, queue);
  }else if(trans_a == 1 && trans_b==1){ // // a->k_a*m, b->n*k, c->k_a*k
    //err = magma_dmalloc (d_c, k_a*k);
    magma_dgemm(MagmaTrans, MagmaTrans, k_a, k, n, alpha, d_a, m, d_b, k, beta,
      *d_c, k_a, queue);
  }else if(trans_a == 1 && trans_b==0){ // a->k_a*m, b->k*n, c->k_a*n
    //err = magma_dmalloc (d_c, k_a*n);
    magma_dgemm(MagmaTrans, MagmaNoTrans, k_a, n, m, alpha, d_a, m, d_b, k,
      beta, *d_c, k_a, queue);
  }else if(trans_a==0 && trans_b==1){ // a->m*k_a, b->n*k, c->m*k
    //err = magma_dmalloc (d_c, m*k);
    magma_dgemm(MagmaNoTrans, MagmaTrans, m, k, n, alpha, d_a, m, d_b, k,
      beta, *d_c, m, queue);
  }
}


double * allocate_identity_device(unsigned m,  magma_queue_t queue ){
  double *h_i, *d_i;
  h_i = (double *) malloc(m*m*sizeof(double)); // Allocate I in host
  magma_dmalloc (&(d_i) , m*m ); // Allocate I in device

  for(int i=0; i<m; i++){
    for(int j=0; j<m; j++){
      if(i==j){
        h_i[i*m + j] = 1.0;
      }
      else{
        h_i[i*m + j] = 0.0;
      }
    }
  }
  // Copy Identity Matrix to the device
  magma_dsetmatrix (m, m, h_i, m, d_i, m, queue);
  free(h_i); /// Free provisional identity matrix in the host
  return d_i;
}

double * calculate_inverse_qr(double *d_a, unsigned m, magma_queue_t queue ){
  // Create Identity Matrix for the inverse
  double *d_i;
  d_i = allocate_identity_device(m, queue);
  if(0){
    double *h_i;
    h_i = malloc(m*m*sizeof(double));
    magma_dgetmatrix(m, m, d_i, m, h_i, m, queue);
    printf("\n Fun :I \n" );
    print_matrix_debug(h_i, m, m);
    free(h_i);
  }
  if(0){
    double *h_a;
    h_a = malloc(m*m*sizeof(double));
    magma_dgetmatrix(m, m, d_a, m, h_a, m, queue);
    printf("\n Fun :AA' \n" );
    print_matrix_debug(h_a, m, m);
    free(h_a);
  }
  // Find the inverse
  magma_int_t iter ;
  magma_int_t info ;
  magma_int_t m_ = m;
  double * d_out;
  magma_dmalloc (&(d_out) , m*m ); // Allocate I in device
  magma_dsgeqrsv_gpu( m_, m_, m_, d_a, m_, d_i, m_, d_out, m_, &iter, &info );
  if(0){
    printf("\n dsgeqrsv status: %d \n", (int) info );
    double *h_i;
    h_i = malloc(m*m*sizeof(double));
    magma_dgetmatrix(m, m, d_out, m, h_i, m, queue);
    printf("\n Fun :(AA')^-1 \n" );
    print_matrix_debug(h_i, m, m);
    free(h_i);
  }
  magma_free(d_i); // We dont need this provisional matrix anymore
  return d_out;
}

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

void matrix_multiplication_device(double *d_a, double *d_b,  unsigned row_a,
  unsigned row_b, unsigned col_b, unsigned col_a, int trans_a, int trans_b, magma_queue_t queue, magma_int_t dev){

  // a->m*k_a(lda m), b->k*n (ldb k)
  // Magma dimension types (it takes into account the matrices are stored in
  // column major fashion or transposed)
  magma_int_t m = row_a;
  magma_int_t k = row_b; // number fo columns of A
  magma_int_t n  = col_b;
  magma_int_t k_a  = col_a;

  magma_int_t err ; // error handler


  // Create constants
  double alpha = MAGMA_S_MAKE ( 1.0 , 0.0 );
  double beta = MAGMA_S_MAKE ( 0.0 , 0.0 );

  double * d_c; // Result Matrix pointer
  if(trans_a == 0 && trans_b==0){ // a->m*k_a, b->k*n, c->m*n
    err = magma_dmalloc (&d_c, m*n);
    magma_dgemm(MagmaNoTrans, MagmaNoTrans, m, n, k, alpha, d_a, m, d_b, k, beta,
      d_c, m, queue);
  }else if(trans_a == 1 && trans_b==1){ // // a->k_a*m, b->n*k, c->k_a*k
    err = magma_dmalloc (&d_c, k_a*k);
    magma_dgemm(MagmaTrans, MagmaTrans, k_a, k, n, alpha, d_a, m, d_b, k, beta,
      d_c, k_a, queue);
  }else if(trans_a == 1 && trans_b==0){ // a->k_a*m, b->k*n, c->k_a*n
    err = magma_dmalloc (&d_c, k_a*n);
    magma_dgemm(MagmaTrans, MagmaNoTrans, k_a, n, m, alpha, d_a, m, d_b, k, beta,
      d_c, k_a, queue);
  }else if(trans_a==0 && trans_b==1){ // a->m*k_a, b->n*k, c->m*k
    err = magma_dmalloc (&d_c, k_a*n);
    magma_dgemm(MagmaNoTrans, MagmaTrans, m, k, k_a, alpha, d_a, m, d_b, k, beta,
      d_c, m, queue);
  }

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
  free(ID);
}

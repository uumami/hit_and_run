/*--------- Standar Libraries-------------*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
/*-------------------------------------- */

/*--------- Algebra libraries--------- */
#include "mkl.h"
#include "mkl_service.h"
#include "i_malloc.h"
/*-------------------------------------- */

/*----- Parameters for setting matrix sizes ------- */
// Number of variables
static unsigned N;
// Number of Equalities restrictions
static unsigned ME;
// Number of Inequalities
static unsigned MI;

/* ------------------------------------------------ */



/*---------- Pointers to restrictions --------------*/

// Matrix of Equalities
static double *E;

// Matrix of Inequalities
static double *W;

// Vector of Equalities
static double *BE;

// Vector of Inequalities
static double *BW;


// Creates pointer P for projection matrix A'(AA')^-1*A, which will later store I- A'(AA')^-1*A
static  double *P

/* ------------------------------------------------ */

int a_at( )
{
  /*
    + A represents a pointer for the mkl routine, but contains matrix E
    + AAT represents a pointer for the output of the mkl routine, contains EE'
    + A = A & C = AAT
    + https://software.intel.com/en-us/mkl-tutorial-c-multiplying-matrices-using-dgemm
    + https://software.intel.com/en-us/node/520775
  */

  // Creates variables necessary for dgemm routine
  //double *A;

  double *AAT;
  int m, n, k;
  double alpha, beta;

  m = ME, n = ME, k =N;
  alpha = 1.0, beta = 0.0;

  // m = 5, n = 5, k =5;
  // double *B;
  // B   = (double *)mkl_malloc( m*k*sizeof( double ), 64 );


  // Allocates memory for the operation
  //A   = (double *)mkl_malloc( m*k*sizeof( double ), 64 );
  AAT = (double *)mkl_malloc( k*n*sizeof( double ), 64 );


  // Checks if memory was allocated succesfully

  // if (A == NULL || AAT == NULL)
  // {
  //   printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
  //   mkl_free(A);
  //   mkl_free(AAT);
  //
  //   // mkl_free(B);
  //
  //   return 1;
  // }

 // Fills the mkl pointer with the equality matrix

  // for (int i = 0; i < m; i++)
  // {
  //   for ( int j = 0; j < k; j++)
  //   {
  //     A[ i*k + j ] = (double)E[i*k + j];
  //
  //     B[ i*k + j ] = (double)E[i][j];
  //
  //
  //
  //     A[ i*k + j ] = (double)0.0;
  //     B[ i*k + j ] = (double)0.0;
  //     if(j == i) A[ i*k + j ] = (double)1.0;
  //     if(i == 0) B[ i*k + j ] = (double)j+1;
  //
  //   }
  // }

 printf("\n-----------E--------------\n" );
 for(int i = 0 ; i < m; i++)
 {
    printf(" \n " );
    for(int j = 0 ; j < k; j++)
    {
      printf(" %lf ",E[i*k+j] );
    }
  }

  // printf("\n-----------B--------------\n" );
  // for(int i = 0 ; i < m; i++)
  // {
  //    printf(" \n ");
  //    for(int j = 0 ; j < k; j++)
  //    {
  //      printf(" %lf ",B[i*k +j] );
  //    }
  //  }

 // Fills the output matrix with zeros
  for (int i = 0; i < (m*n); i++)
  {
    AAT[i] = -1.0;
  }

  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
              m, n, k, alpha, E, k ,E, k, beta, AAT, n);


  // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
  //             m, n, k, alpha, A, k, B, n, beta, AAT, n);

  printf("\n-----------AAT--------------\n" );
    for(int i = 0 ; i < m; i++)
    {
      printf(" \n " );
      for(int j = 0 ; j < n; j++)
      {
        printf(" %lf ",AAT[i*n+j] );
      }
    }



/* Matrix Inversion using MKL Lapack */

  // Creates the identity matrix_read
  double *I;
  I = (double *)mkl_malloc( m*m*sizeof( double ), 64 );

  for (int i = 0; i < m; i++)
  {
    for(int j = 0; j< m; j++)
    {
      if(j==i) {I[i*m+j]=1 ;} else {I[i*m+j]=0;}
    }

  }


  printf("\n-----------I--------------\n" );
  for(int i = 0 ; i < m; i++)
  {
    printf(" \n " );
    for(int j = 0 ; j < m; j++)
    {
      printf(" %lf ",I[i*m+j] );
    }
  }

  int dgels_info = 10;
  dgels_info = LAPACKE_dgels(LAPACK_ROW_MAJOR,'N',m,m,m,AAT,m,I,m);

  if(dgels_info!=0)
  {
    if(dgels_info<0){ printf("\n The %dth parameter has an ilegal value \n",-dgels_info );}else
    { printf("\n The %dth diagonal element of the triangular factor of A is zero, so that A does not have full rank. OLS can't be completed \n",-dgels_info );}
  }else{  mkl_free(AAT);}

  printf("\n-----------AAT inverse--------------\n" );
  for(int i = 0 ; i < m; i++)
  {
    printf(" \n " );
    for(int j = 0 ; j < m; j++)
    {
      printf(" %lf ",I[i*m+j] );
    }
  }

  // Pointer to store the matrix A'(AA')^-1
  double *AC;
  AC = (double *)mkl_malloc( k*m*sizeof( double ), 64 );

  // Fills matrix AC for debug
  for (int i = 0; i < k*m; i++)
  {
    AC[i]= (double) -15555.0;

  }

// Multiplies A'(AA')^-1
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
              k, m, m, alpha, E, k ,I, m, beta, AC, m);

  printf("\n-----------CA--------------\n" );
  for(int i = 0 ; i < k; i++)
  {
    printf(" \n " );
    for(int j = 0 ; j < m; j++)
    {
      printf(" %lf ",AC[i*m+j] );
    }
  }
  // Frees Matrix I
  mkl_free(I);

// Creates pointer I_N for identity matrix of n*n
  double *I_N;

  P   = (double *)mkl_malloc( k*k*sizeof( double ), 64 );
  I_N = (double *)mkl_malloc( k*k*sizeof( double ), 64 );

  for (int i = 0; i < k; i++)
  {
    for(int j = 0; j< k; j++)
    {
      P[i*k+j] = -15555.0;
      if(j==i) {I_N[i*k+j]=1 ;} else {I_N[i*k+j]=0;}
    }

  }

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              k, k, m, alpha, AC, m ,E, k, beta, P, k);

// Frees AC matrix
  mkl_free(AC);

  printf("\n-----------P--------------\n" );
  for(int i = 0 ; i < k; i++)
  {
    printf(" \n " );
    for(int j = 0 ; j < k; j++)
    {
      printf(" %lf ",P[i*k+j] );
    }
  }

  // For : I - A'(AA')⁻1*A. beta=-1 represents the scalar associated with the last term
  beta = -1;
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              k, k, k, alpha, I_N, k ,I_N, k, beta, P, k);

  // Frees I_N
  mkl_free(I_N);

  printf("\n-----------P--------------\n" );
  for(int i = 0 ; i < k; i++)
  {
    printf(" \n " );
    for(int j = 0 ; j < k; j++)
    {
      printf(" %lf ",P[i*k+j] );
    }
  }


  mkl_free(I_N);
  //mkl_free(AAT);
  // mkl_free(A);
  // mkl_free(B);

  return 0;

}

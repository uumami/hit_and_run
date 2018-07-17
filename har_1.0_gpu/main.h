/*--------- Standard Libraries-------------*/
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

/*--------- Algebra libraries--------- */
#include <cuda_runtime.h>
#include "cublas_v2.h"
/*-------------------------------------- */


/* -------------lp_solve library---------*/
#include "lpsolve/lp_lib.h"
/*-------------------------------------- */


/*------------- Our Libraries------------*/
#include "matrix/matrix_read.c"
#include "chebyshev/center.c"
#include "normal/normal_sample.c"
#include "matrix/matrix_operations_mkl.c"
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

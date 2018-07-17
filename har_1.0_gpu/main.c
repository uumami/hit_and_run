
/*------------- Our Libraries------------*/
#include "matrix/matrix_read.c"
#include "chebyshev/center.c"
#include "normal/normal_sample.c"
#include "matrix/matrix_operations_mkl.c"
//#include "matrix/matrix_operations_magma.c"

/*-------------------------------------- */

int main()
{

  /* Counts varaibles and restrictions */
  /* See matrix/matrix_read.c; section Count */
  count();

  /* Reads restiction matrices */
  /* See matrix/matrix_read.c; section Reading Matrices*/
  read_matrices();

  printf("\n --- Matrix Equality ---\n");
  matrix_debug(E,ME,N);

  printf("\n --- Matrix Inequality ---\n");
  matrix_debug(W,MI,N);

  printf("\n --- Vector b Equality ---\n");
  vector_debug(BE,ME);

  printf("\n --- Vector b Inequality ---\n");
  vector_debug(BW,MI);

  // Finds Chebyshev Center of the polytope by linear programming
  double x_0[N];
  get_initvalue(x_0);

  printf("\n --- The center is --- \n");
  vector_debug(x_0,N);

/*
  This section calculates the matrices that will be used in the algorithm,
  so they can stay in the gpu if necessary.
*/

/* --------------------- For double precision -----------------*/

  a_at();


/*------------------- Ends double precision -----------------------*/

  // Initialize varaible for direction d(Nx1) d~ N(0.I).
  double *d;
  d =  (double *)malloc(sizeof(double) * N);

  /*
  * We have the inner point and thus the main loop of the algorithm begins here
  */


  normal_direction(d);

  printf("\n --- d ---\n");
  vector_debug(d,N);

  // Free pointers

  free_restrictions();
  free(d);

  return 1;


}




void matrix_debug(double *q,int m, int n)
{
  for(int i = 0; i <m;i++)
  {
    printf(" %d : ",i);

    for(int j = 0; j <n;j++)
    {
      printf("%lf ",q[i*n + j]);
    }

    printf("\n");
  }

}

void vector_debug(double *q,int m)
{

  for(int i = 0; i <m;i++)
  {
     printf(" %d : %lf \n",i,q[i]);
  }
  printf("\n");

}

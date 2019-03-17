/* ----------------------- Prototype functions ------------------------------ */

/***************************  count_rows *************************************
  * This function allocates the restriction matrices in the GPU
  * Requirements: None
  * Inputs:
      + The separtor between lines (rows) must be different from the varaible
        (column) separator. We recomend using the new line character.
      + Pointer to the txt File.
  * Output:
      + Returns the address where the matrix will live.
*******************************************************************************/
double * allocate_matrices_host(FILE *file, unsigned size);

/* *************************** Print Matrix Debug *************************** */
/*****************************************************************************
* This function prints the matrix or vector living in the pointer P
* Inputs:
  + double *P: Pointer to the matrix
  + unsigned n: Number of count_rows
  + unsigend m: NUmber of columns
* Output:
  + Int of success and prints the matrices
***************************************************************************** */
int print_matrix_debug(double *P, unsigned n, unsigned m);

/***************** allocate_matrices_host_har ********************************
  * This function allocates and saves the matrices in memory.
  * Requirements: allocate_matrices
  * Inputs:
      + None
  * Output:
      + None, the matrices are allocated via the pointers defined in section
      Pointers to restrictions (at the beginning of this header file).
*******************************************************************************/
int allocate_matrices_host_har(int verbose);

/***************** allocate_matrices_host_har ********************************
  * This function allocates and saves the matrices in memory.
  * Requirements: allocate_matrices_host_har
  * Inputs:
      + None
  * Output:
      + Free allocated matrices in host
***************************************************************************** */
int free_host_matrices_har();

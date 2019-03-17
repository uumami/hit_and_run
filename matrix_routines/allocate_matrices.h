/* ----------------------- Prototype functions ------------------------------ */

/***************************  count_rows *************************************
  * This function allocates the restriction matrices in the GPU
  * Requirements: None
  * Inputs:
      + The separtor between lines (rows) must be different from the varaible
        (column) separator. We recomend using the new line character.
      + Pointer to the txt File.
  * Output:
      + Returns and unsigned integer with the number of lines (rows)
      in the txt file.
*******************************************************************************/
int allocate_matrices_host( double *P, FILE *file, unsigned size);

/* *************************** Print Matrix Debug *************************** */

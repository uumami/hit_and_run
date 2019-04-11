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

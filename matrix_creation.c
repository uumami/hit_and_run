#include <stdlib.h>
#include <stdio.h>
#include <string.h>



/*
Create automatic restrictions for polytopes and hyper cubes and save them in in
put matrices
*/

int generate_restrictions(char *type, unsigned dimm){

  FILE *fptr_AE;
  FILE *fptr_AI;
  FILE *fptr_bE;
  FILE *fptr_bI;

  // use appropriate location if you are using MacOS or Linux
  fptr_AE = fopen("input_matrices/A_equality.txt","w");
  fptr_AI = fopen("input_matrices/A_inequality.txt","w");
  fptr_bE = fopen("input_matrices/b_equality.txt","w");
  fptr_bI = fopen("input_matrices/b_inequality.txt","w");

  if((fptr_AE == NULL) | (fptr_AI == NULL) | (fptr_bE == NULL) | (fptr_bE == NULL))
  {
     printf("Error!");
     exit(1);
  }

  if (type=="c"){
  // Only inequality restrictions needed. I<1 and -I<1
    printf("ENtered the loop");

    for(unsigned r=0; r<dimm; r++){
        for(unsigned c=0; c<dimm; c++){
          if(c==r){
            fprintf(fptr_AI,"%lf",1.0);
            fprintf(fptr_bI,"%lf",1.0);
          }else{
            fprintf(fptr_AI,"%lf",0.0);
          }
          if(c != (dimm -1)){fprintf(fptr_AI,"%s"," ");}
        }
        fprintf(fptr_AI,"%s\n","\n");
        fprintf(fptr_bI,"%s\n","\n");
    }

    for(unsigned r=0; r<dimm; r++){
        for(unsigned c=0; c<dimm; c++){
          if(c==r){
            fprintf(fptr_AI,"%lf",-1.0);
            fprintf(fptr_bI,"%lf",1.0);
          }else{
            fprintf(fptr_AI,"%lf",0.0);
          }
          if(c != (dimm -1)){fprintf(fptr_AI,"%s"," ");}
        }
        fprintf(fptr_AI,"%s\n","\n");
        fprintf(fptr_bI,"%s\n","\n");
    }

  }



  // Close all fill_matrix_zeros
     fclose(fptr_AE);
     fclose(fptr_AI);
     fclose(fptr_bE);
     fclose(fptr_bI);
     return 0;
}


void main(){

    char *type = "c";
    unsigned dimm=3;

    generate_restrictions(type, dimm);


}

// header
#include "read_matrix.h"


/****************************  count_rows *************************************/

unsigned count_rows(FILE *file){
// Reading character
  int ch    =   0;
// Flag for new line
  int aux_line   =  1;
// Number of lines
  unsigned lines =   0;

  // Warn about inexisting restrictions
  if(file == NULL){
    printf(" \n The program can not find the files containing the\
    restrictions in the folder input_matrices \n ");
  }

// Parsing the file
  while ((ch = fgetc(file)) != EOF){
    if (ch == '\n')         {aux_line = 1;}

  // 48 and 57 represents the ascii code for 0 and 9 respectibly
    if (ch <= 57 && ch >=48 && aux_line)  {lines++, aux_line=0;}
  }
  return lines;
}


/***************************  count_columns ***********************************/
unsigned count_columns(FILE *file)
{
// Reading character
  int ch    =   0;
//  Flag of new variable
  int aux_space  =  1;
// Number of vars
  unsigned vars  =   0;

// Debug for inexistent restrictions
if(file == NULL){
  printf(" \n The program can not find the files containing the\
  restrictions in the folder input_matrices \n ");
}

// Parsing the file to get the number of variables
  while ((ch = fgetc(file)) != EOF && ch != '\n')
  {
    if (ch == ' '){ // Values separeted by spaces
      aux_space = 1;
    } else if (ch == ','){ // Values separated by commas
      aux_space = 1;
    }
    // 48 and 57 represents the ascii code for 0 and 9 respectibly
    if (ch <= 57 && ch >=48 && aux_space)  {vars++, aux_space=0;}
  }
  return vars;
}


/***************************  count_columns ***********************************/
int parse_restrictions(){
  // Aux varaible for identifying the number of restrictions
  unsigned NE = 0;
  unsigned NI = 0;

  // Pointers to txt files
  FILE *a_eq    = fopen(AE_TXT, "r");
  FILE *a_in    = fopen(AI_TXT, "r");
  FILE *b_eq    = fopen(bE_TXT, "r");
  FILE *b_in    = fopen(bI_TXT, "r");

  // Parse Equality vector for rows
  ME = count_rows(b_eq);
  fclose(b_eq);
  printf("Number of (rows) in the Equality Vector: %u \n", ME );
  // Parse Inequality vector for rows
  MI = count_rows(b_in);
  fclose(b_in);
  printf("Number of (rows) in the Inequality Vector: %u \n", MI );

  // Parse the number of variables (columns) in AE
  NE       = count_columns(a_eq);
  fclose(a_eq);
  // Parse the number of variables (columns) in AI
  NI = count_columns(a_in);
  fclose(a_in);
  // pRINT
  printf("Number of variables (columns) in the Equality Matrix is: %u \n", NE );
  printf("Number of variables (columns) in the Inquality Matrix is: %u \n", NI );
  N = NE;
  // This condition is only tiggered if some matrix is empty
  if(NI != NE){
    printf(" THE NUMBER OF VARIABLES IN EACH MATRIX DIFFERS. WE WILL \
    TAKE THE NUMBER OF VARIABLES (COLUMNS) IN THE EQUALITY RESTRICTIONS \
    AS %u.\n", N);
  }

  printf("Number of Variables: %u \n", N );
  return 1;
}

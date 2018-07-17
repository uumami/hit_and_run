/*--------- Standar Libraries-------------*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
/*-------------------------------------- */


/* -------------lp_solve library---------*/
#include "../lpsolve/lp_lib.h"
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

/* ------------------------------------------------ */


/*
  * Computes norm 2 of a vector
  + Input is the size of the vector and the vector.
  + Returns the norm 2 of the vector
*/

double norm_2(int N, double *x, int r) {
  double sum = 0.0;
  for (int i = 0; i < N; i++) {
    sum += pow(x[ r*N + i], 2.0);
  }
  return sqrt(sum);
}
/* ------------------------------------------------ */



void get_initvalue(double x_0[N]) {
  lprec *lp;
  lp = make_lp(0, 0);

  set_col_name(lp, (N+1), "R");

  // Objective function
  double obj_vector[N + 2];

  for (int i = 0; i < (N + 1); i++)
  {
    obj_vector[i] = 0.0;
  }

  obj_vector[(N + 1)] = 1.0;

  // Set objective function
  set_obj_fn(lp, obj_vector);
  set_add_rowmode(lp, TRUE);

  // Set equality restrictions
  double aux[N + 2];

  for (int i = 0; i < ME; i++)
  {
    for (int j = 0; j < N; j++)
    {
      aux[j + 1] = (E[i*N + j]);
    }

    // ||a_i||_2
    aux[(N + 1)] = 0;
    add_constraint(lp, aux, EQ, BE[i]);
  }

    // Set inequality restrictions
  for (int i = 0; i < MI; i++)
  {
    for (int j = 0; j < N; j++)
    {
      aux[j + 1] = (W[i*N + j]);
    }

    // ||a_i||_2
    aux[(N + 1)] = norm_2(N, W, i);
    add_constraint(lp, aux, LE, BW[i]);
  }

  // Positivity conditions
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
    {
      if(i == j)
      {
        aux[j+1] = 1.0;
      } else
      {
        aux[j+1] = 0.0;
      }
    }

    aux[(N + 1)] = -1;
    add_constraint(lp, aux, GE, 0);
  }

  // End model definition
  set_add_rowmode(lp, FALSE);
  set_maxim(lp);

  // write_lp(lp,"lp.model");
  solve(lp);

  get_variables(lp, x_0);
}

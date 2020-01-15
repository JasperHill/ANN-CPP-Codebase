///////////////////////////////////////////
// Node.h
// 10/21/2018 - J. Hill
//////////////////////////////////////////

#ifndef _NODE_H
#define _NODE_H

#include "Base.h"

//////////////////////////////////////////

struct Node
{
  string type;          //node type (tanh, pi, sigma)

  int input_size;
  gsl_vector* input;   //input vector - to be stored for backpropagation-based updates
  gsl_vector* w;       //input weights
  gsl_vector* dw;      //corrections to the weights
  gsl_vector* dFdw;    //gradient with respect to the input weights
  gsl_vector* dFdB;    //gradient with respect to the node bias
  gsl_vector* grad_x;  //gradient with respect to inputs
  double B;            //node bias

};


void Initialize(string s,int n, double b);
void ResetW(int n);
void InitializeW(gsl_vector* v);
double f(gsl_vector* x);
double dfdy(void);
gsl_vector* dfdw(void);                   //gives the gradient with respect to w
double dfdB(void);                        //gives the gradient with respect to B
gsl_vector* dfdx(void);                   //gives the gradient with respect to input
void Update(double alpha, string mode);
void FreeMemory(void);

#endif

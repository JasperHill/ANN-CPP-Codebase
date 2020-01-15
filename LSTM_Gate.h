/////////////////////////////////////////////////////
// LSTM_Gate.h
// 10/22/2018 - J. Hill
/////////////////////////////////////////////////////

#ifndef _LSTM_GATE_H
#define _LSTM_GATE_H

#include "Base.h"
#include "Node.h"

/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
//Layout
//
//   0      1    2    3     4
//0 =i------π----∑----π-----o
//          |    |\   |
//1 =i------|    π<|  |
//               |    |
//2 =i-----------|    |
//                    |
//3 =i----------------|
//
//

struct LSTM_Gate
{
  const int K = 1;

  double pi01_i0;
  double pi01_i1;

  double sigma02_i0;
  double sigma02_i1;

  double pi12_i0;
  double pi12_i1;//cell_state

  double pi03_i0;//cell_state
  double pi03_i1;

  double cell_state;

  int num_of_nodes;
  Node** Inputs;
  Node* Output;

  LSMT_Gate(void){num_of_nodes = 4; Inputs = (Node**) malloc(4*sizeof(Node*));}
};

void Initialize(vector<string> s, gsl_vector* w, double b);
double f(gsl_vector* x);
gsl_vector* dfdw_i(int i);                       //gives the gadient with respect to input weights for node i
double dfdb_i(int i);                            //gives the gradient with respect to input biases for node i
gsl_vector* dfdx_i(int i);
void ComputeCorrections(double C, double eta);
void Update(double alpha, string mode);                           //allows for stochastic or deterministic update via alpha
void Free(void);

#endif

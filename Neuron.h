/////////////////////////////////////////////////////////////////////////////
// Neuron.h
// 10/23/2018 - J. Hill
/////////////////////////////////////////////////////////////////////////////

#ifndef _GATE_H
#define _GATE_H

#include "Base.h"
#include "LSTM_Gate.h"

/////////////////////////////////////////////////////////////////////////////

struct Neuron
{
  int InputSize; //number of input nodes
  string type;
  void* pGate;

  int inputsize = 0; //number of inputnodes
  int receives_training_data;
  gsl_vector* f;
  gsl_vector* dfdx;
};

void Initialize(string s, std::vector<string> types, void* g, gsl_vector* w, double b);
void ResetInputSize(int n);
void ConnectToTrainingManifold(int n);
double f_i(gsl_vector* training_x, gsl_vector* layer_x, gsl_vector* params);
void Learn(double C, double eta);
double dfdx_i(double C, int i);                                       //returns the gradient with respect to input vectors
void Update(double alpha, string mode);
void Die(void);

#endif

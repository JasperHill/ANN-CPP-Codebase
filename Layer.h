/////////////////////////////////////////////////////
// Layer.h
// 10/23/2018 - J. Hill
//////////////////////////////////////////////////////

#ifndef _LAYER_H
#define _LAYER_H

#include "Base.h"
#include "Neuron.h"


//////////////////////////////////////////////////////

struct Layer
{
  Neuron** Neurs;//array of neurons
  int size;//number of neurons
  gsl_vector* func;//output vector
  int initiated;

  Layer(void){size = 0; initiated = 0;};
};

void AddNeurons(int n, Neuron** Neurons);
void Initiate(int n, Neuron** Neurons);
void f(gsl_vector* training_x, gsl_vector* layer_x, gsl_vector* params, Neuron** Neurs);
void TeachNeurons(vector<double> C, double alpha);
void Update(double alpha, string mode);

#endif

///////////////////////////////////////////////////////////////////
// NeuralNet.h
// 10/23/2018 - J. Hill
///////////////////////////////////////////////////////////////////

#ifndef _NEURALNET_H
#define _NEURALNET_H

#include "Base.h"
#include "Layer.h"

///////////////////////////////////////////////////////////////////
// Neurons are created individually and collected into layers
// Layers are added to the network
// The highest layer index corresponds to the network output while 0
//    corresponds to the input
///////////////////////////////////////////////////////////////////
struct NeuralNet
{
  Neuron*** Neurons;                //a 2D array of pointers to each neuron in the network -- Neurons[Layer index][position within layer]
  Layer** Layers;                   //collections of indices corresponding to neurons with the same inter-neuronal connections
  vector< vector<int> > Connections;//matrix of layer indices [(input_0,output_0),(input_1,output_1),...(input_N-1,output_N-1)] for N connections
  int Nparams;                      //number of free parameters to be optimized

  int D;                    //number of layers in network
  double wS;                //entropy weight factor
  double S;                 //a measure of the network's entropy S = ∑ wS*(w_i + b_i) for weights and biases
  double E;                 //the squared network error E = ∑ (1/2)(f_i - y_i)^2
  double F;                 //the value of the objective function -- F = wS*S + E
  int input_size;           //size of the training data
  gsl_vector* Output;

  NeuralNet(int depth){D = depth; Connections.resize(0); Layers = (Layer**) malloc(D*sizeof(Layer*));}
};
//neurons are created and arranged externally

void AddLayer(Layer* pL, Neuron* pN, int i);
void ConnectLayers(in Source, int Destination);                 //connects two layers via their memory addresses
void CreateManifold(Layer* Destination);                        //connects a layer with the training data
void Initiate(void);
int Index_from_NetworkAddress(int Li, int Ni);                  //returns the gsl_vector index corresponding to the Ni_th neuron of the Li_th layer
vector<int> NetworkAddress_from_Index(int index);               //returns the Neurons indices corresponding to the gsl_vector index
void Propagate(gsl_vector* x, gsl_vector* params);              //here, x is the training input, and params contains weight and bias information
void BackPropagate(vector<double> C);               //C is a vector of modifiers for each network output // BackPropagate can also be used in SA methods
                                                                //to propagate thermal perturbations
void Update(double alpha, string mode);                         //passes update information and style (deterministic or stochastic) backward through the network
void CalcError(gsl_vector* Target, gsl_vector* Input);          //calculates E
void CalcEntropy(void);                                         //calculates S
void CalcF(void);                                               //calculates F

#endif

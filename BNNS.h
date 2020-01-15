////////////////////////////////////////////////////////////////////////////////
// BNNS.h
// 10/29/2018 - J. Hill
////////////////////////////////////////////////////////////////////////////////

#ifndef _BNNS_H
#define _BNNS_H

#include "Base.h"
#include "NeuralNet.h"
#include "NN_Simplex_Geometry.h"

struct Brownian_NN_Simplex
{
  NeuralNet* ProtoBrain;           //a neuralnet pointer containing layout information for the target neuralnet
  gsl_vector** TrainingSet;        //the time series to be learned
  int T;                           //number of points in the time series

  int NBrains;                     //number of neural nets in the simplex =/= number of points to optimize
  int Nparams;                     //number of parameters to be optimized ~T^3 for market data of size T
  NeuralNet* SmartestBrain;        //the optimum simplex point
  vector<int> BrainIndices;        //array of Neural Net indices with the first index corresponding to the best net and the last corresponding to the worst
  gsl_vector** Vs;      //array of simplex coordinates

  double tau;              //temperature of the simplex//initially 1
  double d;                    //relative spread between maximum and minimum network errors // d = 2|f_max - f_min|/(f_max + f_min)
  double min_d;                //minimum spread below which the simplex is considered fully annealed
  double f_min;
  double f_max;

  //Brownian_NN_Simplex(NeuralNet* pB, gsl_vector** TS, int t){ProtoBrain = pB; TrainingSet = TS; T = t;}
};

void Initialize(NeuralNet* pB, gsl_vector** TS, int t);
void CountParamaters(void);
void StoreParameters(int source_i);   //stores the paramaters of Vs[source_i] in ProtoBrain for propagation/backpropagation
void ExtractParameters(int dest_i);   //extracts parameters from ProtoBrain and stores them in Vs[dest_i]
void CreateSimplexHyperspace(void);
void Bang(void);//produces a random scattering of the Nparams+1 Neural Nets across the objective surface at a temperature 10*tau
void Reorder(void);
void Perturb(NeuralNet* pBrain, double sign);
void ReflectWorst(void);//reflects worst point of simplex about the hyperplane formed by all other points
void ExpandIn1D(void);//extends worst point outward away from simplex hyperplane
void ContractIn1D(void);//contracts worst point toward simplex hyperplane
void ContractAroundMin(void);//contracts all inferior points toward optimum point
void Anneal(void);

#endif

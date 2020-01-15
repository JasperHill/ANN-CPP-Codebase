////////////////////////////////////////////////////////////////////////////
//  AUX.h
//  05/23/2019 - J. Hill
////////////////////////////////////////////////////////////////////////////

#ifndef _AUX_H
#define _AUX_H

#include "Base.h"
#include "NeuralNet.h"

////////////////////////////////////////////////////////////////////////////

void TransferParametersToVector(NeuralNet* N, vector<double>& v)
{
  int size = 0;
  int NiMax,kmax,lmax;

  Layer* pL;

  for (int Li = 0; Li < N->D; Li++)
    {
      NiMax = N->Layers[Li]->size - 1;
      pL->Layers[Li];

      for (int Ni = 0; Ni <= NiMax; Ni++)
	{
	  kmax = ((N->Neurons[Li][Ni])->pGate)->num_of_nodes;

	  for (int k = 0; k <= kmax; k++)
	    {

	    }
	}
    }


  return;
}

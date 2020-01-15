///////////////////////////////////////////////////////////////
//  NN_Simplex_Geometry.h
//  5/21/2019 - J. Hill
///////////////////////////////////////////////////////////////

#ifndef _NN_SIMPLEX_GEOMETRY_H
#define _NN_SIMPLEX_GEOMETRY_H

#include "Base.h"
#include "NeuralNet.h"

///////////////////////////////////////////////////////////////


struct NN_Simplex_Coordinate
{
  //int dim;              //dimension of the hyperspace occupied by the simplex
  NeuralNet* pBrain;      //
  gsl_vector* V;          //vector containing every parameter in pBrain to be optimized
  double Magnitude;       //

  NN_Simplex_Coordinate(NeuralNet* pN){dim = 0; pBrain = pN;}
};


///////////////////////////////////////////////////////////////

void CalcDimension(void);
void Initialize(NeuralNet* pN);                     //sets pN = pBrain and initializes V
void InitializeV(void);                             //extracts network parameters from pBrain and places them in V
void Magnitude(NN_Simplex_Coordinate* pC);
void Update_pBrain(void);                           //projects the parameters of V onto pBrain in the correct order
void BePerturbed(double c);                         //applies a gaussian perturbation, scaled by c, to each element of V
void Scale(double c);
void Subtract(NN_Simplex_Coordinate* pCi, NN_Simplex_Coordinate* pCj, NN_Simplex_Coordinate* pCk);
double Dot(NeuralNet* pNi, NeuralNet* Nj);
void Free(NN_Simplex_Coordinate* pC);

#endif

///////////////////////////////////////////////////////////////
//  NN_Simplex_Geometry.C
//  5/22/2019 - J. Hill
///////////////////////////////////////////////////////////////

#include "NN_Simplex_Geometry.h"

///////////////////////////////////////////////////////////////

void CalcDimension(void)
{
  int index = 0;

  for (int Li = 0; Li < pBrain->D; Li++)
    {
      for (int dd = 0; dd < (pBrain->Layers[Li])->size; dd++)
	{
	  wi = (((pBrain->Neurons[Li][dd])->pGate)->Output)->w;
	  index += wi->size;

	  kmax = (pBrain->Neurons[dd])->InputSize - 1;
      
	  for (int k = 0; k <= kmax; k++)
	    {
	      wi = (((pBrain->Neurons[Li][dd])->pGate)->Inputs[k])->w;

	      index += wi->size;
	    }//k
	}//dd
    }//Li


  dim = index+1;
  V = gsl_vector_alloc(dim);
  return;
}

int ValidIndex(int i)
{
  int imax = dim-1;

  if (i <= imax)
    {
      return 1;
    }

  else
    {
      cout<<"Invalid index: i = "<<i<<"; imax = "<<imax<<endl;
      return 0;
    }
}

void StoreParamaters(void)
{
  gsl_vector* wi;
  int ddmax,kmax,lmax,index;
  index = 0;

  for (int Li = 0; Li < pBrain->D; Li++)
    {
      for (int dd = 0; dd < (pBrain->Layers[Li])->size; dd++)
	{
	  wi = (((pBrain->Neurons[Li][dd])->pGate)->Output)->w;
	  gsl_vector_set(V,index,(((pBrain->Neurons[Li][dd])->pGate)->Output)->B);
	  index++;

	  if !(ValidIndex(index)) break;

	  for (int k = 0; k < wi->size; k++)
	    {
	      gsl_vector_set(V,index,gsl_vector_get(wi,k));
	      index++;

	      if !(ValidIndex(index)) break;
	    }

	  kmax = (pBrain->Neurons[dd])->InputSize - 1;
      
	  for (int k = 0; k <= kmax; k++)
	    {
	      lmax = ((((pBrain->Neurons[Li][dd])->pGate)->Inputs[k])->w)->size - 1;
	      wi = (((pBrain->Neurons[Li][dd])->pGate)->Inputs[k])->w;
	      gsl_vector_set(V,index,(((pBrain->Neurons[Li][dd])->pGate)->Inputs[k])->B);
	      index++;
	      
	      if !(ValidIndex(index)) break;

	      for (int l = 0; l <= lmax; l++)
		{
		  gsl_vector_set(V,index,gsl_vector_get(wi,l));
		  index++;

		  if !(ValidIndex(index)) break;
		}//l
	    }//k
	}//dd
    }//Li

  return;
}//end of StoreParamaters

void ExtractParameters(void)
{
  if (Nparams != v->size)
    {
      cout<<"Error: vector v is of invalid size"<<endl;
    }

  gsl_vector* wi;
  int ddmax,kmax,lmax,index,imax;
  imax = Nparams-1;
  index = 0;

  for (int Li = 0; Li < pN->D; Li++)
    {
      for (int dd = 0; dd < (pN->Layers[Li])->size; dd++)
	{
	  newB = gsl_vector_get(v,index);
	  (((pN->Neurons[Li][dd])->pGate)->Output)->B = newB;
	  index++;

	  if !(ValidIndex(index,imax)) break;

	  wi = (((pN->Neurons[Li][dd])->pGate)->Output)->w;

	  for (int k = 0; k < wi->size; k++)
	    {
	      gsl_vector_set(wi,gsl_vector_get(v,k));
	      index++;

	      if !(ValidIndex(index,imax)) break;
	    }

	  kmax = (pN->Neurons[dd])->InputSize - 1;
      
	  for (int k = 0; k <= kmax; k++)
	    {
	      newB = gsl_vector_get(v,index);
	      (((pN->Neurons[Li][dd])->pGate)->Inputs[k])->B = newB;
	      index++;
	      
	      if !(ValidIndex(index,imax)) break;

	      wi = (((pN->Neurons[Li][dd])->pGate)->Inputs[k])->w;
	      lmax = wi->size - 1;

	      for (int l = 0; l <= lmax; l++)
		{
		  gsl_vector_set(wi,l,index,gsl_vector_get(v,l));
		  index++;

		  if !(ValidIndex(index,imax)) break;
		}//l
	    }//k
	}//dd
    }//Li

  return;
}//end of ExtractParamaters

void Scale(double c, NeuralNet* pN)
{
  ///////////////////////////////////////////////////////////////////////////////////////////
  //
  //scales the weight vectors and node biases of NeuralNet N and stores the information in N_prime
  //
  ///////////////////////////////////////////////////////////////////////////////////////////

  for (int Li = 0; Li < pN->D; Li++)
    {

    }

  return;
}

double Dot(NeuralNet* pNi, NeuralNet* pNj)
{
  ///////////////////////////////////////////////////////////////////////////////////////////
  //
  //computes the hypergeometric dot product of NeuralNets Ni and Nj containing d neurons
  //
  ///////////////////////////////////////////////////////////////////////////////////////////

  if (pNi->D != pNj->D)
    {
      cout<<"Dot product not possible; neural networks are of different depth"<<end;
      return 0;
    }

  double c = 0;
  gsl_vector* wi;
  gsl_vector* wj;
  double Bi,Bj;

  for (int Li = 0; Li < pNi->D; Li++)
    {
      for (int dd = 0; dd < (pNi->Layers[Li])->size; dd++)
	{
	  wi = (((pNi->Neurons[Li][dd])->pGate)->Output)->w;
	  wj = (((pNj->Neurons[Li][dd])->pGate)->Output)->w;

	  Bi = (((pNi->Neurons[Li][dd])->pGate)->Output)->B;
	  Bj = (((pNj->Neurons[Li][dd])->pGate)->Output)->B;

	  c += Bi*Bj;

	  for (int k = 0; k < wi->size; k++)
	    {
	      c += gsl_vector_get(wi,k)*gsl_vector_get(wj,k);
	    }

	  kmax = (pNi->Neurons[dd])->InputSize - 1;
      
	  for (int k = 0; k <= kmax; k++)
	    {
	      lmax = ((((pNi->Neurons[Li][dd])->pGate)->Inputs[k])->w)->size - 1;
	      
	      wi = (((pNi->Neurons[Li][dd])->pGate)->Inputs[k])->w;
	      wj = (((pNj->Neurons[Li][dd])->pGate)->Inputs[k])->w;
	      
	      Bi = (((pNi->Neurons[Li][dd])->pGate)->Inputs[k])->B;
	      Bj = (((pNj->Neurons[Li][dd])->pGate)->Inputs[k])->B;
	      
	      c += Bi*Bj;
	      
	      for (int l = 0; l <= lmax; l++)
		{
		  c += gsl_vector_get(wi,l)*gsl_vector_get(wj,l);
		}//l
	    }//k
	}//dd
    }//Li
  return c;
}

double Magnitude(NeuralNet* N, int d)
{
  double c = 0;

  gsl_vector* v;
  double b;

  for (int Li = 0; Li < N->D; Li++)
    {
      for (int dd = 0; dd < (N->Layers[Li])->size; dd++)
	{
	  v = (((N->Neurons[Li][dd])->pGate)->Output)->w;
	  b = (((N->Neurons[Li][dd])->pGate)->Output)->B;

	  c += gsl_pow_2(b);

	  for (int k = 0; k < v->size; k++)
	    {
	      c += gsl_pow_2(gsl_vector_get(v,k));
	    }

	  kmax = (Ni->Neurons[Li][dd])->InputSize - 1;
	  
	  for (int k = 0; k <= kmax; k++)
	    {
	      lmax = ((((Ni->Neurons[Li][dd])->pGate)->Inputs[k])->w)->size - 1;

	      v = (((N->Neurons[Li][dd])->pGate)->Inputs[k])->w;
	      b = (((N->Neurons[Li][dd])->pGate)->Inputs[k])->B;

	      c += gsl_pow_2(b);

	      for (int l = 0; l <= lmax; l++)
		{
		  c += gsl_pow_2(gsl_vector_get(v,l));
		}//l
	    }//k
	}//dd
    }//Li

  return c;
}

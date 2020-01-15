///////////////////////////////////////////////////////////
// BNNS.C
// 10/30/2018 - J. Hill
///////////////////////////////////////////////////////////

#include "BNNS.h"

///////////////////////////////////////////////////////////

void Brownian_NN_Simplex::Initialize(NeuralNet* pB, gsl_vector** TS, int t)
{
  ///////////////////////////////////////////////////////////////////////////
  //  Sets the neural network, layout sets the training data, creates the
  //  vector of network indices, and counts the simplex space dimension
  ///////////////////////////////////////////////////////////////////////////

  ProtoBrain = pB;
  TrainingSet = TS;
  T = t;
  tau = 1;

  BrainIndices.resize(T);

  for (int i = 0; i < T; i++)
    {
      BrainIndices[i] = i;
    }

  CountParameters();
  NBrains = Nparams+1;
  BrainIndices.resize(NBrains);

  for (int i = 0; i <= Nparams; i++)
    {
      BrainIndices[i] = i;
    }

  Vs = (gsl_vector**) malloc(NBrains * sizeof(gsl_vector*));

  return;
}//end of Initialize

void Brownian_NN_Simplex::CountParamters(void)
{
  ///////////////////////////////////////////////////////////////////////////
  //  a function to count all free parameters in the neural network layout
  //  
  ///////////////////////////////////////////////////////////////////////////

  int LiMax,NiMax,kmax;
  Nparams = 0;

  LiMax = ProtoBrain->D - 1;

  for (int Li = 0; Li < LiMax; Li++)
    {
      NiMax = (ProtoBrain->Layers[Li])->size - 1;

      for (int Ni = 0; Ni <= NiMax; Ni++)
	{
	  Nparams++;
	  Nparams += ((((ProtoBrain->Neurons[Li][Ni])->pGate)->Output)->w)->size;

	  kmax = ((ProtoBrain->Neurons[Li][Ni])->pGate)->num_of_nodes - 1;

	  for (int k = 0; k <= kmax; k++)
	    {
	      Nparams++;
	      Nparams += ((((ProtoBrain->Neurons[Li][Ni])->pGate)->Inputs[k])->w)->size;
	    }//k
	}//Ni
    }//Li

  return;
}


void Brownian_NN_Simplex::StoreParamters(int source_i)
{
  ///////////////////////////////////////////////////////////////////////////
  //  Stores the paramaters of Vs[source_i] in ProtoBrain
  //  
  ///////////////////////////////////////////////////////////////////////////

  int LiMax,NiMax,kmax;
  int index = 0;
  gsl_vector* v = Vs[source_i];

  LiMax = ProtoBrain->D - 1;

  for (int Li = 0; Li < LiMax; Li++)
    {
      NiMax = (ProtoBrain->Layers[Li])->size - 1;

      for (int Ni = 0; Ni <= NiMax; Ni++)
	{
	  gsl_vector_set(v,index,(((ProtoBrain->Neurons[Li][Ni])->pGate)->Output)->B);
	  index++;
	  Nparams += ((((ProtoBrain->Neurons[Li][Ni])->pGate)->Output)->w)->size;

	  kmax = ((ProtoBrain->Neurons[Li][Ni])->pGate)->num_of_nodes - 1;

	  for (int k = 0; k <= kmax; k++)
	    {
	      Nparams++;
	      Nparams += ((((ProtoBrain->Neurons[Li][Ni])->pGate)->Inputs[k])->w)->size;
	    }//k
	}//Ni
    }//Li

  return;
}


void Brownian_NN_Simplex::Bang(void)
{
  ///////////////////////////////////////////////////////////////////////////
  //  Creates Nparams sets of network parameters randomly displaced from
  //  the set obtained from ProtoBrain (Vs[0]) and stores them in Vs[1,...,Nparams]
  //
  ///////////////////////////////////////////////////////////////////////////

  int index = 0;
  double b,weight;
  gsl_vector* wi;

  gsl_rng* r = gsl_rng_alloc(gsl_rng_default);
  clock_t time = clock();
  gsl_rng_set(r,time);

  LiMax = ProtoBrain->D - 1;
  Vs[0] = gsl_vector_alloc(Nparams);

  for (int Li = 0; Li < LiMax; Li++)
    {
      NiMax = (ProtoBrain->Layers[Li])->size - 1;
      
      for (int Ni = 0; Ni <= NiMax; Ni++)
	{
	  kmax = ((((ProtoBrain->Neurons[Li][Ni])->pGate)->Output)->w)->size - 1;
	  wi = (((ProtoBrain->Neurons[Li][Ni])->pGate)->Output)->w;
	  b = (((ProtoBrain->Neurons[Li][Ni])->pGate)->Output)->B;
	  gsl_vector_set(Vs[0],index,b);
	  index++;
	  
	  for (int k = 0; k <= kmax; k++)
	    {
	      gsl_vector_set(Vs[0],index,gsl_vector_get(wi,k));
	      index++;
	    }
	  
	  kmax = ((ProtoBrain->Neurons[Li][Ni])->pGate)->num_of_nodes - 1;
	  
	  for (int k = 0; k <= kmax; k++)
	    {
	      lmax = ((((ProtoBrain->Neurons[Li][Ni])->pGate)->Inputs[k])->w)->size - 1;
	      wi = (((ProtoBrain->Neurons[Li][Ni])->pGate)->Inputs[k])->w;
	      
	      b = (((ProtoBrain->Neurons[Li][Ni])->pGate)->Inputs[k])->B;
	      gsl_vector_set(Vs[0],index,b);
	      index++;
	      
	      for (int l = 0; l <= lmax; l++)
		{
		  gsl_vector_set(Vs[0],index,gsl_vector_get(wi,l));
		  index++;
		}//l
	    }//k
	}//Ni
    }//Li


  double b0,w0,std_dev;
  gsl_rng* r = gsl_rng_alloc(gsl_rng_default);
  clock_t time = clock();
  gsl_rng_set(r,time);
  std_dev = 1;

for (int i = 1; i <= Nparams)
  {
    Vs[i] = gsl_vector_alloc(Nparams);

    for (int j = 0; j < Nparams; j++)
      {
	gsl_vector_set(Vs[i],j,gsl_vector_get(Vs[0],j) + tau*gsl_ran_gaussian(r,std_dev));
      }
  }

  return;
}//end of Bang

void Brownian_NN_Simplex::Perturb(NeuralNet* pBrain, double sign)
{
  double alpha;
  gsl_rng* r = gsl_rng_alloc(gsl_rng_default);
  clock_t time = clock();
  gsl_rng_set(r,time);
  alpha = log(gsl_ran_flat(r,Base::ZERO,1));
  alpha *= sign*tau;

  Layer* pL;

  for (int Li = 0; Li < D; Li++)
    {
      pL = (pBrain->Layers)[i];
      pL->Update(alpha,"stochastic");
    }

  return;
}//end of Perturb


void Reorder(void)
{
  NeuralNet* pTempNet;
  double worst_f = -10008; 
  double best_f = 10008;
  int worst_i,best_i;
  double f;


  for (int i = 0; i < NBrains; i++)
    {
      f = 0;

      for (int t = 0; t < T; t++)
	{
	  Brains[i]->CalcF(TS[t+1],TS[t]);
	  f += Brains[i]->F;
	}

      if (f > worst_f)
	{
	  worst_f = f;
	  worst_i = i;
	}

      else if (i == 0 || f < best_f)
	{
	  best_f = f;
	  best_i = i;
	}
    }

  pTempNet = Brains[NBrains-1];
  Brains[NBrains-1] = Brains[worst_i];
  Brains[worst_i] = pTemptNet;

  pTempNet = Brains[0];
  Brains[0] = Brains[best_i];
  Brains[best_i] = pTemptNet;

  return;
}//end of Reorder

void ReflectWorst(void)
{
  NeuralNet* Ni,Nw;

  Nw = Brains[NBrains-1];



  return;
}

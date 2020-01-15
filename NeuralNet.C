/////////////////////////////////////////////////////////////////////
// NeuralNet.C
// 10/23/2018 - J. Hill
/////////////////////////////////////////////////////////////////////

#include "NeuralNet.h"

void NeuralNet::AddLayer(Layer* pL, Neuron* pN, int i)
{
  if (i > D-1)
    {
      cout<<"Invalid layer index specified"<<endl;
    }

  else
    {
      Layers[i] = pL;
      Neurons[i] = pL->Neurs;
    }

  return;
}

void NeuralNet::ConnectLayers(int Source, int Destination)
{
  int n;
  vector<int> Connection(2);
  Connection[0] = Source;
  Connection[1] = Destination;
  Connections.push_back(Connection);
  n = (Layers[Source])->size;

  for (int i = 0; i < (Layers[Destination])->size; i++)
    {
      (Desination->Neurs)[i]->ResetInputSize(n);
    }

  return;
}

void NeuralNet::CreateManifold(Layer* Destination)
{
  for (int i = 0; i < Destination->size; i++)
    {
      (Destination->Neurs[i])->ConnectToTrainingManifold(input_size);
    }

  return;
}

void NeuralNet::Initiate(void)
{
  Output = gsl_vector_alloc(Connections[Connections.size()-1][1]->OutputSize);
  return;
}

int NeuralNet::Index_from_NetworkAddress(int Li, int Ni, int k, int l, int is_input, int is_B)
{
  /////////////////////////////////////////////////////////////
  //  returns gsl_vector index corresponding to network address
  //
  /////////////////////////////////////////////////////////////

  int index = 0;
  int kmax,lmax;

  for (int Lj = 0; Lj < Li; Lj++)
    {
      for (int Nj = 0; Nj < Ni; Nj++)
	{
	  kmax = ((Neurons[Li][Ni])->pGate)->num_of_nodes - 1;
	  index++;
	  index += (((Neurons[Li][Ni]->pGate)->Output)->w)->size;

	  for (int kk = 0; kk <= kmax; kk++)
	    {
	      index++;
	      index += (((Neurons[Li][Ni]->pGate)->Inputs[k])->w)->size;
	    }
	}
    }

  if is_input {index += (((1+Neurons[Li][Ni]->pGate)->Output)->size) + (is_B + (!is_B)*l);}

  else {index += is_B + (!is_B)*l;}

  return index;
}

vector<int> NeuralNet::NetworkAddress_from_Index(int index)
{
  /////////////////////////////////////////////////////////////
  //  returns network address corresponding to gsl_vector index
  //
  /////////////////////////////////////////////////////////////

  int Li,Ni,k,l,is_input,is_B;
  int iindex = 0;
  int NiMax,kmax,lmax;

  for (int Lj = 0; Lj < D; Lj++)
    {
      NiMax = Layers[Lj]->size - 1;

      for (int Nj = 0; Nj <= NiMax; Nj++)
	{
	  kmax = (Neurons[Lj][Nj]->pGate)->num_of_nodes - 1;

	  for (int kk = 0; kk <= kmax; kk++)
	    {

	    }
	}
    }

  return Address;
}

void NeuralNet::CountParameters(void)
{
  Nparams = 0;
  int NiMax,kmax;

  for (int Li = 0; Li < D; Li++)
    {
      NiMax = Layers[Li]->size - 1;

      for (int Ni = 0; Ni <= NiMax; Ni++)
	{
	  Nparams++;
	  Nparams += (((Neurons[Li][Ni]->pGate)->Output)->w)->size;

	  kmax = (Neurons[Li][Ni]->pGate)->num_of_nodes - 1;

	  for (int k = 0; k <= kmax; k++)
	    {
	      Nparams++;
	      Nparams += ((((Neurons[Li][Ni])->pGate)->Inputs[k])->w)->size;
	    }//k
	}//Ni
    }//Li
}

/*
int NeuralNet::Index(int Li, int Ni)
{
  //retreives the index for neuron Ni in layer Li
  int index = 0;

  for (int i = 0; i < Li; i++)
    {
      index += Layers[i]->size;
    }

  index += Ni;

  return index;
}
*/

void NeuralNet::Propagate(gsl_vector* x, gsl_vector* params)
{
  //propagates the training input x
  vector<double> temp_vecdoub;
  gsl_vector* temp_layer_x,temp_output,output;
  int status;
  int i1,i2;

  i1 = Connections[0][0];
  temp_layer_x = gsl_calloc((Neurons[i1]->size)+1);

  for (int i = 0; i < Connections.size(); i++)
    {
      i1 = Connections[i][0];
      i2 = Connections[i][1];
      Neurons[i1]->f(x,temp_layer_x);
      Neurons[i2]->f(x,Neurons[i1]->func);

      gsl_vector_free(temp_layer_x);
      temp_layer_x = gsl_vector_alloc(Neurons[i2]->size);
      status = gsl_vector_memcpy(temp_layer_x,Nerons[i2]->func);

      if (status != GSL_SUCCESS)
	{
	  cout<<"Error in data propagation at layer "<<i<<endl;
	}
    }

  return;
}

void NeuralNet::BackPropagate(vector<double> C)//propagates weight update information backward through the network for gradient-based learning algorithms
{

  gsl_matrix* grad;
  double delta_grad_i;
  vector<double> c;//learning modifier for each node of each lower layer
  vector<double> C_prime = C;
  Neuron* pNeuron;
  Node** pNodeArray;
  int jj,kmax,lmax;

  for (int Li = D; Li >= 1; Li--)
    {
      Layers[Li]->TeachNeurons(C_prime,eta);
      c.resize(0);
      c.resize(Layers[Li-1]->size,0);
      kmax = c.size()-1;
      grad = gsl_vector_calloc(kmax+1);

      for (int Ni = 0; Ni < Layers[Li]->size; Ni++)
	{
	  //jj = Index(Li,j);
	  pNodeArray = (Neurons[Li][Ni]->pGate)->Inputs;
	  
	  for (int k = 0; k <= kmax; k++)//sums over the neurons of each layer
	    {
	      delta_grad_i = Neurons[Li][Ni]->dfdx_i(C_prime[j],k);
	      gsl_vector_set(grad,k,gsl_vector_get(grad,k)+delta_grad_i);
	      c[k] += gsl_vector_get(grad,k);
	    }//k
	}//j
	       
      gsl_vector_free(grad);
      C_prime = c;
    }//Li
  
  Layers[0]->TeachNeurons(C_prime);

  ~c();
  ~C_prime();
  gsl_matrix_free(grad);

  return;
}

void NeuralNet::Update(double alpha, string mode)
{
  for (int Li = D; Li >= 0; i--)
    {
      Layers[Li]->Update(alpha,mode);
    }

  return;
}

void NeuralNet::CalcError(gsl_vector* Target, gsl_vector* Input)
{
  double e;

  if (target->size != Output->size)
    {
      cout<<"Error: target vector is of invalid dimension"<<endl;
    }

  Propagate(Input);
  e = 0;

  for (int i = 0; i < Output->size; i++)
    {
      e += gsl_pow_2(gsl_vector_get(Output,i) - gsl_vector_get(target,i));
    }

  e *= 0.5;
  E = e;

  return;
}

void NeuralNet::CalcEntropy(void)
{
  double s = 0;
  int NiMax;
  Node** pNodeArray;

  for (int Li = 0; Li < D; Li++)
    {
      NiMax = Layers[Li]->size;

      for (int Ni = 0; Ni < NiMax; Ni++)
	{
	  kmax = (Neurons[Li][Ni]->InputSize)-1;
	  pNodeArray = ((Neurons[Li][Ni])->pGate)->Inputs;

	  for (int k = 0; k <= kmax; k++)
	    {
	      s += pNodeArray[k]->B;

	      for (int l = 0; l < (pNodeArray[k]->w)->size; l++)
		{
		  s += gsl_vector_get(pNodeArray[k]->w,l);
		}
	    }
	}
    }

  S = s;

  return;
}

void NeuralNet::CalcF(gsl_vector* Target, gsl_vector* Input)
{
  CalcError(Target,Input);
  CalcEntropy();
  F = E + wS*S;

  return;
}

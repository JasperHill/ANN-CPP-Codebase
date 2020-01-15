///////////////////////////////////////////////////////////////////////
// Layer.C
// 10/23/2018 - J. Hill
///////////////////////////////////////////////////////////////////////

#include "Layer.h"

///////////////////////////////////////////////////////////////////////

void Layer::AddNeurons(int n, Neuron** Neurons)
{
  size = n;
  Neurs = Neurons;
  return;
}

void Layer::Initiate(int n, Neuron** Neurons)
{
  AddNeurons(n,Neurons);
  initiated = 1;
  return;
}

void Layer::CreateManifold(int n)
{
  if (initiated == 0)
    {
      cout<<"Cannot create manifold; layer not initiated"<<endl;
      return;
    }

  for (int i = 0; i < size; i++)
    {
      Neurs[i]->ConnectToTrainingManifold(n);
    }

  return;
}

void Layer::f(gsl_vector* training_x, gsl_vector* layer_x)
{
  //training_x is the vector of training data // the individual neurons know whether or not to accept this data
  //layer_x is the output from the previous layer of neurons
  //Neurs is the array of neurons whose outputs form func

  if (initiated == 0)
    {
      cout<<"Cannot compute output; layer not initiated"<<endl;
      return;
    }

  for (int i = 0; i < size; i++)
    {
      gsl_vector_set(func,i,Neurs[i]->f_x(training_x,layer_x));
    }

  return;
}

void Layer::TeachNeurons(vector<double> C)
{
  if (initiated == 0)
    {
      cout<<"Cannot teach neurons; layer not initiated"<<endl;
      return;
    }

  for(int i = 0; i < size; i++)
    {
      Neurs[i]->Learn(C[i]);
    }

  return;
}

void Layer::Update(double alpha, string mode)
{
  if (initiated == 0)
    {
      cout<<"Cannot update node weights; layer not initiated"<<endl;
      return;
    }

  for (int i = 0; i < size; i++)
    {
      Neurs[i]->Update(alpha,mode);
    }

  return;
}

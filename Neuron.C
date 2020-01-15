///////////////////////////////////////////////////////////////////////////
// Neuron.C 
// 10/23/2018 - J. Hill
///////////////////////////////////////////////////////////////////////////

#include "Neuron.h"

///////////////////////////////////////////////////////////////////////////

void Neuron::Initialize(int manifold, string s, std::vector<string> types, void* g, gsl_vector* w, double b)
{
  if (manifold == 0 || manifold == 1)
    {
      receives_training_data = manifold;
      type = s;

      if (type == "LSTM")
	{
	  pGate = (LSTM_Gate*) g;
	  pGate->Initialize(types,w,b);
	}
    }

  else
    {
      cout<<"Invalid manifold parameter specified"<<endl;
    }

  return;
}

void Neuron::ResetInputSize(int n)
{
  gsl_vector* old_w;
  gsl_vector* new_w;

  inputsize += n;

  if (type == "LSTM")
    {
      for (int i = 0; i < (pGate->Inputs).size(); i++)
	{
	  old_w = (pGate->Inputs)[i]->W;
	  new_w = gsl_vector_alloc(old_w->size + n);
      
	  for (int j = 0; j < new_w->size; j++)
	    {
	      if (j < old_w->size)
		{
		  gsl_vector_set(new_w,j,gsl_vector_get(old_w,j));
		}
	      
	      else
		{
		  gsl_vector_set(new_w,j,0);
		}
	    }//j

	  (pGate->Inputs)[i].ResetW(new_w);
	  gsl_vector_free(new_w);
	}//i
    }
  return;
}
void Neuron::ConnectToTrainingManifold(int n)
{
  if (receives_training_data == 1)
    {
      cout<<"Neuron already connected to training manifold"<<endl;
      return;
    }

  receives_training_data = 1;
  ResetInputSize(n);

  return;
}


gsl_vector* Neuron::f(gsl_vector* training_x, gsl_vector* layer_x)
{
  gsl_vector* v;
  input = x;

  if (type == "LSTM")
    {
      v = gsl_vector_alloc(1);
      gsl_vector_set(v,0,pGate->f(training_x,layer_x));
    }

  return v;
}

void Neuron::Learn(double C)
{
  pGate->ComputeCorrections(C);
  return;
}

double Neuron::dfdx_i(double C, int i)
{
  //////////////////////////////////////////////////////////////////
  //
  //returns the gradient with respect to the input of the ith input node
  //
  //////////////////////////////////////////////////////////////////
  double grad_i;

  if (type == "LSTM")
    {
      grad = pGate->dfdx_i(i);
      gsl_vector_scale(grad,C);
    }

  else
    {
      gsl_vector_set_zero(grad);
    }

  return grad;
}

void Neuron::Update(void)
{
  pGate->Update();
  return;
}

void Neuron::Die(void)
{
  pGate->Free();
  return;
}

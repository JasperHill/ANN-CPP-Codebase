/////////////////////////////////////////////////
// LSTM_Gate.C
// 10/22/2018 - J. Hill
/////////////////////////////////////////////////

#include "LSTM_Gate.h"

/////////////////////////////////////////////////

void LSTM_Gate::Initialize(vector<string> s, gsl_vector* w, double b)
{

  if (s.size() >= 4)
    {
      for (int i = 0; i < 4; i++)
	{
	  Inputs[i].Initialize(s[i],w,b);
	}

      if (s.size() == 4)
	{
	  Output.Initialize(s[s.size()-1],w,b);
	}

      else
	{
	  gsl_vector* v = gsl_vector_alloc(1);
	  gsl_vector_set(v,0,1);
	  Output.Initialize(s[4],v,b);

	  gsl_vector_free(v);
	}
    }

  else if (s.size() < 4)
    {
      for (int i = 0; i < s.size(); i++)
	{
	  Inputs[i].Initialize(s[i],w,b);
	}

      for (int j = s.size(); j < 4; j++)
	{
	  Inputs[j].Initialize(s[s.size()-1],w,b);
	}

      gsl_vector* v = gsl_vector_alloc(1);
      gsl_vector_set(v,0,1);
      Output.Initialize(s[s.size()-1],v,b);

      gsl_vector_free(v);
    }

  cell_state = 0;

  return;
}

double LSTM_Gate::f(gsl_vector* training_x, gsl_vector* layer_x)
{
  gsl_vector* Output_x = gsl_vector_alloc(1);

  /////////////////////////////////////////////////////
  //compute inputs for first pi node

  pi01_i0 = Inputs[0].f(training_x,layer_x);
  pi01_i1 = Inputs[1].f(training_x,layer_x);

  ///////////////////////////////////////////////////////
  //compute inputs for second pi node

  pi12_i0 = Inputs[2].f(training_x,layer_x);
  pi12_i1 = cell_state;

  ///////////////////////////////////////////////////////
  //compute inputs for memory cell

  sigma02_i0 = pi01_i0*pi01_i1;
  sigma02_i1 = pi12_i0*pi12_i1;

  ///////////////////////////////////////////////////////
  //compute inputs for third pi node

  pi03_i0 = sigma02_i0 + sigma02_i1;
  cell_state = pi03_i0;
  pi03_i1 = Inputs[3].f(training_x,layer_x);

  ///////////////////////////////////////////////////////
  //compute inputs for output node

  gsl_vector_set(Output_x,0,(pi03_i0*pi03_i1));
  y = Output.f(Output_x);

  return y;
}

gsl_vector* LSTM_Gate::dfdw_i(int i)
{
  ////////////////////////////////////////////////////////////////
  //
  //returns the weight gradient of the ith input node
  //
  ////////////////////////////////////////////////////////////////

  double C;
  gsl_vector* grad = gsl_vector_alloc((Inputs[i]->w)->size);
  C = Output.dfdx(0);

  if (i == 0 || i == 1)
    {
      C *= pi03_i1;

      if (i == 0)
	{
	  C *= pi01_i1;
	}

      else if (i == 1)
	{
	  C *= pi01_i0;
	}
    }  

  else if (i == 2)
    {
      C *= pi03_i1;
      C *= pi12_i1;
    }

  else if (i == 3)
    {
      C *= pi03_i0;
    }  


  gsl_vector_memcpy(grad,Inputs[i]->dfdw());
  gsl_vector_scale(grad,C);
  return grad;
}

gsl_vector* LSTM_Gate::dfdx_i(int i)
{
  ////////////////////////////////////////////////////////////////
  //
  //returns the gradient with respect to the ith input node
  //
  ////////////////////////////////////////////////////////////////
  double C;
  gsl_vector* grad;

  C = gsl_vector_get(Output->dfdx,0);

  if (i == 0 || i == 1)
    {
      C *= pi03_i1;

      if (i == 0)
	{
	  C *= pi01_i1;
	}

      else if (i == 1)
	{
	  C *= pi01_i0;
	}
    }  

  else if (i == 2)
    {
      C *= pi03_i1;
      C *= pi12_i1;
    }

  else if (i == 3)
    {
      C *= pi03_i0;
    }  

  gsl_memcpy(grad,Inputs[i]->dfdx());
  gsl_vector_scale(grad,C);
  return grad;
}

void LSTM_Gate::ComputeCorrections(double C)
{
  gsl_vector_memcpy(Output->dw,Output->dfdw());
  gsl_vector_scale(Output->dw,C);

  gsl_vector_memcpy(Inputs[0]->dw,dfdw_i(0));
  gsl_vector_memcpy(Inputs[1]->dw,dfdw_i(1));
  gsl_vector_memcpy(Inputs[2]->dw,dfdw_i(2));
  gsl_vector_memcpy(Inputs[3]->dw,dfdw_i(3));

  return;
}

void LSTM_Gate::Update(double alpha, string mode)
{
  Output.Update(alpha,   mode);
  Inputs[0].Update(alpha,mode);
  Inputs[1].Update(alpha,mode);
  Inputs[2].Update(alpha,mode);
  Inputs[3].Update(alpha,mode);

  return;
}

void LSTM_Gate::Free(void)
{
  Output.FreeMemory();
  Inputs[0].FreeMemory();
  Inputs[1].FreeMemory();
  Inputs[2].FreeMemory();
  Inputs[3].FreeMemory();
  ~Inputs();
  return;
}

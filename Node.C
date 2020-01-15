//////////////////////////////////////////////////////////////////
// Node.C
// 10/22/2018 - J. Hill
//////////////////////////////////////////////////////////////////

#include "Node.h"

void Node::Initialize(string s, int n, double b)
{
  type = s;
  w = gsl_vector_calloc(n);
  input_size = n;

  B = b;

  return;
}

void Node::ResetW(gsl_vector* new_w)
{
  input_size = new_w->size;
  gsl_vector_free(W);
  W = gsl_vector_alloc(N);
  gsl_vector_memcpy(W,new_w);

  return;
}

void Node::InitializeW(gsl_vector* v)
{
  if (w->size != v->size)
    {
      cout<<"Error in initialization of node weights"<<endl;
    }

  gsl_vector_memcpy(w,v);
  return;
}

double Node::f(gsl_vector* training_x, gsl_vector* layer_x)
{
  gsl_vector* x = gsl_vector_alloc(N);
  int offset;

  if (N == layer_x->size)
    {
      gsl_vector_memcpy(x,layer_x);
    }

  else if (N == (training_x->size)+(layer_x->size))
    {
      offset = training_x->size;

      for (int i = 0; i < training_x->size; i++)
	{
	  gsl_vector_set(x,i,gsl_vector_get(training_x,i));
	}

      for (int j = 0; j < layer_x->size; j++)
	{
	  gsl_vector_set(x,j+offset,gsl_vector_get(layer_x,j));
	}
    }

  else
    {
      cout<<"Input vector incompatible with weight vector"<<endl;
      return 0;
    }

  double y = 0;

  if (type == "tanh" || type == "sigma")
    {
      for (int i = 0; i < N; i++)
        {
          y += gsl_vector_get(w,i)*gsl_vector_get(x,i);
        }

      if (type == "tanh")
	{
	  return tanh(y + B);
	}

      else if (type == "sigma")
        {
          return (y + B);
        }
    }


  else if (type == "pi")
    {
      for (int i = 0; i < N; i++)
        {
          y *= gsl_vector_get(x,i);
	}

      return (y + B);
    }

  return (y + B);
}

double dfdy(void)
{
  double a = 0;

  if (type == "tanh")
    {
      a = B;

      for (int j = 0; j < N; j++)
	{
	  a += gsl_vector_get(w,j)*gsl_vector_get(input,j);
	}

      a = gsl_pow_2(sec(a));
      return a;
    }

  else
    {
      cout<<"No type specified. Returning 0."<<endl;
      return 0;
    }

  return a;
}

gsl_vector* Node::dfdw(void)
{
  if (x->size != N)
    {
      cout<<"Input vector incompatible with weight vector"<<endl;

      return 0;
    }

  double a;
  double grad_i;
  gsl_vector* grad = gsl_vector_alloc(N);

  if (type == "tanh")
    {
      a = dfdy();

      for (int i = 0; i < N; i++)
	{     
	  grad_i = a*gsl_vector_get(input,i);
	  gsl_vector_set(grad,i,grad_i);
	}

      return grad;
    }

  else
    {
      cout<<"No type specified. Returning 0."<<endl;
      gsl_vector_set_zero(grad);
    }

  return grad;
}

double Node::dfdB(void)
{
  if (x->size != N)
    {
      cout<<"Input vector incompatible with weight vector"<<endl;

      return 0;
    }

  if (type == "tanh")
    {
      return dfdy();
    }

  else
    {
      cout<<"No type specified. Returning 0."<<endl;
      return 0;
    }
}

gsl_vector* Node::dfdx(void)
{
  if (x->size != N)
    {
      cout<<"Input vector incompatible with weight vector"<<endl;

      return 0;
    }

  double a;
  double grad_i;
  gsl_vector* grad = alloc(N);

  if (type == "tanh")
    {
      a = dfdy;

      for (int i = 0; i < N; i++)
	{
	  grad_i = a*gsl_vector_get(input,i);
	  gsl_vector_set(grad,i,grad_i);
	}

      return grad;
    }

  else if (type == "pi")
    {
      for (int i = 0; i < N; i++)
	{
	  a = 1;

	  for (int j = 0; j < N; j++)
	    {
	      if (j == i)
		{
		  continue;
		}
	      
	      a *= gsl_vector_get(input,j);
	    }

	  gsl_vector_set(grad,a);
	}

      return grad;
    }

  else if (type == "sigma")
    {
      gsl_vector_set_all(grad,1);
      return grad;
    }
}

void Node::Update(double alpha, string mode)
{
  if (mode == "stochastic")
    {
      gsl_vector* v = gsl_vector_alloc(dw->size);
      gsl_vector_set_all(v,alpha);
      gsl_vector_div(v,dw);
      gsl_vector_memcpy(dw,v);
    }

  else if (mode == "deterministic")
    {
      gsl_vector_scale(dw,alpha);
    }

  for (int i = 0; i < N; i++)
    {
      gsl_vector_set(w,i,gsl_vector_get(w,i)+gsl_vector_get(dw,i));
    }

  return;
}

void Node::FreeMemory(void)
{
  gsl_vector_free(w);
  gsl_vector_free(dw);

  return;
}

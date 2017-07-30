//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/Statistics/dsp/Romberg.h

#ifndef __RombergMethod
#define __RombergMethod

#include "Neville.h"
#include "MidPoint.h"
#include "Trapezoid.h"
#include "Error.h"

template< template<typename,typename> class Choose = Trapezoid >
class Romberg {

public:

  float precision;
  bool absolute;

  Romberg () { precision = 1.0e-12; absolute = true; }

  template<typename T, typename Unary>
  T operator () (Unary func, T x1, T x2)
    {
      Choose<Unary,T> choose;

      const unsigned extrapolate = 5;
      const unsigned iterations = 18;
      
      T h[iterations+1];
      T s[iterations];
      
      h[0]=1.0;
      T ss, dss;
      for (unsigned j=0; j<iterations; j++) {

        s[j]= choose(func,x1,x2,j);

        if (j >= extrapolate) {
          Neville (h+j-extrapolate,s+j-extrapolate,extrapolate,0.0,ss,dss);
          if (absolute) {
            if (fabs(dss) <= precision) return ss;
          }
          else if (fabs(dss) <= precision*fabs(ss)) return ss;
        }
  
        h[j+1]=h[j]/choose.order_squared;
  
      }
      throw Error (InvalidState, "Romberg", "maximum iterations %u exceeded;"
                   " delta = %f x1=%lf x2=%lf", iterations, float(dss/ss), x1, x2);

    }

};

//! Performs the change of variable from x to t=1/x in the integrand
template<class Unary, class T>
class OneOverBase {
public:
  typedef T result_type;
  typedef T argument_type;

  OneOverBase (Unary f) : func(f) { }
  T operator () (T arg)
    { return - func(1.0/arg) / (arg*arg); }
protected:
  Unary func;
};

//! Returns a functor that changes the variable from func(x) to func(1/x)
template<class T>
OneOverBase<T(*)(T),T> OneOver (T(*func)(T))
{
  return OneOverBase<T(*)(T),T> (func);
}

// F must be an adaptable unary function
template<class F>
OneOverBase<F, typename F::argument_type> OneOver (F function)
{
  return OneOverBase<F,typename F::argument_type> (function);
}

//! Performs the change of variable from x to t=1/x in the integrand
template<class Unary, class T>
class MinusBase {
public:
  typedef T result_type;
  typedef T argument_type;

  MinusBase (Unary f) : func(f) { }
  T operator () (T arg)
    { return -func(-arg); }
protected:
  Unary func;
};

//! Returns a functor that changes the variable from func(x) to func(1/x)
template<class T>
MinusBase<T(*)(T),T> Minus (T(*func)(T))
{
  return MinusBase<T(*)(T),T> (func);
}

// F must be an adaptable unary function
template<class F>
MinusBase<F, typename F::argument_type> Minus (F function)
{
  return MinusBase<F,typename F::argument_type> (function);
}

#endif


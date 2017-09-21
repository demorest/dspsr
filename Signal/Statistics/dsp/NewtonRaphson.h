//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/Statistics/dsp/NewtonRaphson.h

#ifndef __NewtonRaphsonMethod
#define __NewtonRaphsonMethod

#include "Error.h"

class NewtonRaphson
{
  public:

  unsigned max_iterations;
  double precision;
  double lower_limit;
  double upper_limit;

  NewtonRaphson ()
  { 
    max_iterations = 100;
    precision = 1e-12;
    lower_limit = 0;
    upper_limit = 0;
  }
  
  double sqr (double x) { return x*x; }

  template<typename Unary, typename UnaryDerivative>
  double operator () (Unary f, UnaryDerivative df, double y, double guess_x)
  {
    for (unsigned i=0; i < max_iterations; i++)
    {
      double dx = (f(guess_x)-y) / df(guess_x);
      guess_x -= dx;
      if (lower_limit != upper_limit)
      {
        if (guess_x < lower_limit)
          guess_x = lower_limit;
        if (guess_x > upper_limit)
          guess_x = upper_limit;
      }
      if (sqr(dx) <= sqr(guess_x*precision))
        return guess_x;
    }
    throw Error (InvalidState, "NewtonRaphson::invert",
      "maximum iterations exceeded; guess=%lf", guess_x);
  }

};


#endif

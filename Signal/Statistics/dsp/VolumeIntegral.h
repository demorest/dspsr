//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/Statistics/dsp/VolumeIntegral.h

#ifndef __Volume_Integral
#define __Volume_Integral

#include "Romberg.h"

static double infinity = 1e100;

//! Computes an integral over a rectangular volume of arbitrary dimension
template<unsigned N, class T, class F>
class VolumeIntegral {

 protected:

  T min[N];
  T max[N];

  F* function;

 public:

  //! set the range of the ith dimension
  void set_range (unsigned i, T _min, T _max) { min[i]=_min; max[i]=_max; }

  //! set the function to be integrated
  void set_function (F* f) { function = f; }

  //! evaluate the function over the specified volume
  double evaluate () { return evaluate (0); }

 protected:

  class Marginal {
  public:
    Marginal (VolumeIntegral* _boss=0, unsigned _i=0) { boss = _boss; i = _i; }

    T operator () (T x) {
      boss->function->set_param (i, x);
      T result;
      if (i==N-1)
	result = boss->function->evaluate();
      else
	result = boss->evaluate(i+1);
      // cerr << i << " " << result << endl;
      return result;
    }
  protected:
    VolumeIntegral* boss;
    unsigned i;
  };

  double evaluate (unsigned i) {

    Romberg<> romberg;
    // romberg.precision = 10e-6 / pow(10., double(i));

    if (min[i] != -infinity && max[i] != infinity)
       return romberg (Marginal(this,i), min[i], max[i]);

    // if one of the range is infinite, must use an open method
    Romberg<MidPoint> r2;
    r2.precision = romberg.precision;

    if (min[i] == -infinity)
      return r2 (OneOverBase<Marginal,T>(Marginal(this,i)), 0., 1/max[i]);
    else if (max[i] == infinity)
      return r2 (OneOverBase<Marginal,T>(Marginal(this,i)), 1/min[i], 0.);

  }
 
};

#endif

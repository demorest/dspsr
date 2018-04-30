//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/Statistics/dsp/Trapezoid.h

#ifndef __TrapezoidMethod
#define __TrapezoidMethod

template<class Unary, class T>
class Trapezoid {
  public:

  T order_squared;

  Trapezoid () { order_squared = 4.0; }

  T operator () (Unary func, T a, T b, unsigned n) {

    if (n == 0) {
      s = 0.5 * (b-a) * (func(a) + func(b));
      return s;
    }

    unsigned it = 1;
    unsigned j = 0;
    
    for (j=1; j<n; j++) it <<= 1;
    
    T tnm=it;
    T del=(b-a)/tnm;
    T x=a+0.5*del;
    T sum=0.0;

    for (j=0; j<it; j++) {
      sum += func(x);
      x += del;
    }
    
    s=0.5*(s+(b-a)*sum/tnm);
    
    return s;
  }

  protected:
    T s;

};


#endif

//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/Statistics/dsp/MidPoint.h

#ifndef __MidPointMethod
#define __MidPointMethod

template<class Unary, class T>
class MidPoint {
  public:

  T order_squared;

  MidPoint () { order_squared = 9.0; }

  T operator () (Unary func, T a, T b, unsigned n) {

    if (n == 0) {
      s = (b-a) * func (0.5*(a+b));
      return s;
    }

    unsigned it = 1;
    unsigned j = 0;
    
    for (j=1; j<n; j++) it *= 3;
    
    T tnm=it;
    T del=(b-a)/(3.0*tnm);
    T ddel=del+del;
    T x=a+0.5*del;
    T sum=0.0;

    for (j=0; j<it; j++) {
      sum += func(x);
      x += ddel;
      sum += func(x);
      x += del;
    }
    
    s=(s+(b-a)*sum/tnm)/3.0;
    
    return s;
  }

  protected:
    T s;

};


#endif

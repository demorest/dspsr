//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/Statistics/dsp/Neville.h,v $
   $Revision: 1.2 $
   $Date: 2011/08/04 21:03:22 $
   $Author: straten $ */

#ifndef __NevilleMethod
#define __NevilleMethod

#include <assert.h>
#include <math.h>
#include <vector>

template<typename T>
void Neville (T* xa, T* ya, unsigned n, T x, T& y, T& dy)
{
  unsigned i,m,ns=0;

  std::vector<T> c (n);
  std::vector<T> d (n);

  T dif = fabs (x-xa[0]);

  for (unsigned i=0; i<n; i++) {
    T dift = fabs(x-xa[i]);
    if ( dift < dif) {
      ns=i;
      dif=dift;
    }
    c[i]=ya[i];
    d[i]=ya[i];
  }

  y=ya[ns];
  ns --;

  dy = 0;

  for (unsigned m=1; m<n-1; m++) {
    for (unsigned i=0;i<n-m; i++) {
      T ho=xa[i]-x;
      T hp=xa[i+m]-x;
      T w=c[i+1]-d[i];
      T den = ho - hp;
      assert (den != 0.0);
      den=w/den;
      d[i]=hp*den;
      c[i]=ho*den;
    }
    if (2*ns >= (n-m)){
      dy = d[ns];
      ns --;
    }
    else
      dy = c[ns+1];

    y += dy;
  }
}

#endif

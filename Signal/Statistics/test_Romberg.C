/***************************************************************************
 *
 *   Copyright (C) 2011 by Andrew Jameson and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Romberg.h"

#include <iostream>

using namespace std;

double normal (double x)
{
  return 2/sqrt(M_PI) * exp(-x*x);
}

int main () try
{
  double c0 = erf (1.);
  cerr << "std C says erf(1)=" << c0 << endl;

  Romberg<> romberg;

  // integrate the normal distribution from 0 to 1
  double c1 = romberg (normal, 0., 1.);
  cerr << "Romberg says erf(1)=" << c1 << endl;

  if (fabs(c0-c1) > romberg.precision * fabs(c0) ) {
    cerr << "Error erf(1)=" << c0 << " != Romberg=" << c1 << endl;
    return -1;
  }

  // integrate the normal distribution from 1 to infinity
  Romberg<MidPoint> r2;
  double c2 = r2 (OneOver(normal), 1., 0.);

  cerr << "Expect  1 to inf = " << 1-c0 << endl;
  cerr << "Romberg 1 to inf = " << c2 << endl;

  if (fabs(1-c0-c2) > romberg.precision * fabs(c2) ) {
    cerr << "Error 1-erf(1)=" << 1-c0 << " != Romberg=" << c2 << endl;
    return -1;
  }

  // integrate the normal distribution from negative infinity to -1
  double c3 = r2 (OneOver(Minus(normal)), 0., 1.);

  cerr << "Romber -inf to -1 = " << c3 << endl;

  if (fabs(c3-c2) > romberg.precision * fabs(c2) ) {
    cerr << "Error Romberg(-)=" << c3 << " != Romberg(+)=" << c2 << endl;
    return -1;
  }

  return 0;
}
catch (Error& error) {
  cerr << error << endl;
  return -1;
}


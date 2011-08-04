/***************************************************************************
 *
 *   Copyright (C) 2011 by Andrew Jameson and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "NewtonRaphson.h"

#include <cmath>
#include <iostream>
using namespace std;

// derivative of erf(x) with respect to x
double derf (double x)
{
  return 2/sqrt(M_PI) * exp(-x*x);
}

int main ()
{
  NewtonRaphson invert;

  //
  double y = 0.6;
  double xguess = 1.0;

  double x = invert(erf, derf, y, xguess);

  cerr << "y=" << y << " x=" << x << " erf(x)=" << erf(x) << endl;

  return 0;
}


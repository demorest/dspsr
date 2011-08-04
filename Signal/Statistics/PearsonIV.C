/***************************************************************************
 *
 *   Copyright (C) 2011 by Andrew Jameson and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/PearsonIV.h"

#include <iostream>

#include <math.h>
#include <float.h>

using namespace std;

dsp::PearsonIV::PearsonIV (unsigned M)
{
  SK_M = M;
  verbose = 0;
  prepare ();
}

dsp::PearsonIV::~PearsonIV ()
{
}

void dsp::PearsonIV::prepare ()
{

  double M = (double) SK_M;

  // the mean of the distribution is always 1
  mu1 = 1;

  mu2 = (4 * M * M) / ((M-1) * (M+2) * (M+3));

  beta1  = (4 * (M+2) * (M+3) * (5*M - 7) * (5*M - 7));
  beta1 /= ((M-1) * (M+4) * (M+4) * (M+5) * (M+5));

  beta2 = 3 * (M+2) * (M+3) * (M*M*M + 98*M*M - 185*M + 78);
  beta2 /= ((M-1) * (M+4) * (M+5) * (M+6) * (M+7));

  if (verbose)
    cerr << "PearsonIV::prepare mu1=" << mu1 << " mu2=" << mu2 
         << " beta1=" << beta1 << " beta2=" << beta2 << endl;

  r = (6 * (beta2  - beta1 - 1)) / (2*beta2 - 3*beta1 - 6);

  m = (r+2) / 2;

  v = -1 * (r * (r-2) * sqrt(beta1));
  v /= sqrt(16 * (r-1) - beta1 * (r-2) * (r-2));

  a = 0.25 * sqrt(mu2 * ((16 * (r-1)) - (beta1 * (r-2) * (r-2))));

  lamda = mu1 - 0.25 * (r-2) * sqrt(mu2) * sqrt(beta1);

  if (verbose)
    cerr << "dsp::PearsonIV::prepare r=" << r << " m=" << m << " v=" << v
         << " a=" << a << " lamda=" << lamda << endl;

  // some quick checks
  if (!finite(v))
    cerr << "dsp::PearsonIV::prepare v is not finite M=" << M << endl;

  if (!finite(a))
    cerr << "dsp::PearsonIV::prepare a is not finite M=" << M << endl;

  if (!finite(lamda))
    cerr << "dsp::PearsonIV::prepare lamda is not finite M=" << M << endl;

  if (m <= 0.5)
    cerr << "dsp::PearsonIV::prepare m <= 0.5 M=" << M << endl;

  // calculate the pearson type IV normalisation for these
  // parameters
  logk = log_normal();

  if (!finite(logk))
    cerr << "dsp::PearsonIV::prepare logk not finite" << endl;

}

/*
double dsp::PearsonIV::normalisation ()
{
  double x = m;
  const double y = 0.5 * v;
  const double y2 = y * y;
  const double  xmin = (2*y2>10.0) ? 2*y2 : 10.0;

  double r=1, s=1, p=1, f=0;

  while (x < xmin) 
  {
    const double t = y/x++;
    r *= 1 + t*t;
  }

  while (p > s*DBL_EPSILON) {
    p *= y2 + f*f;
    p /= x++ * ++f;
    s += p;
  }

  double gammar2 = 1.0 / (r*s);

  cerr << "normalisation: gammar2=" << gammar2 << " r=" << r << " s=" << s << endl;

  double k = 0.5 * M_2_SQRTPI * gammar2 * exp (lgamma(m)-lgamma(m-0.5)) / a;

  return k;
}
*/

double dsp::PearsonIV::log_normal ()
{
  double x = m;
  const double y = 0.5 * v;
  const double y2 = y * y;
  const double  xmin = (2*y2>10.0) ? 2*y2 : 10.0;
  
  double logr=0, s=1, p=1, f=0;
  
  while (x < xmin)
  {
    const double t = y/x++;
    logr += log(1 + t*t);
  }
  
  while (p > s*DBL_EPSILON) {
    p *= y2 + f*f;
    p /= x++ * ++f;
    s += p;
  } 
 
  return log (0.5 * M_2_SQRTPI/a) - (logr + log(s)) + lgamma(m) - lgamma(m-0.5);
}


double dsp::PearsonIV::operator() (double x)
{
  double a1 = (x-lamda) / a;

  double a2 = 1 + pow(a1,2);
  double a3 = v * atan (a1);
  
  double res = exp( logk -1 * m * log (a2) - a3);

  //double arg = exp( logk -1 * m * log (1 + pow(((x-lamda)/a),2)) - (v * atan((x-lamda)/a)));

  return res;
}

double dsp::PearsonIV::cf (double x)
{
  if (x < -1)
    return tricky(Minus(OneOver(*this)), 0., -1/x);
  else
    return tricky(Minus(OneOver(*this)), 0., 1.) + normal(*this, -1., x);
}

double dsp::PearsonIV::ccf (double x)
{
  if (x > 1)
    return tricky(OneOver(*this), 1/x, 0.);
  else
    return tricky(OneOver(*this), 1., 0.) + normal(*this, x, 1.);
}

double dsp::PearsonIV::log_cf (double x)
{
  return log(cf(x));
}

double dsp::PearsonIV::dlog_cf (double x)
{
  return (*this)(x) / cf(x);
}


double dsp::PearsonIV::log_ccf (double x)
{
  return log(ccf(x));
}

double dsp::PearsonIV::dlog_ccf (double x)
{
  return -1 * (*this)(x) / ccf(x);
}



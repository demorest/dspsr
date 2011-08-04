/***************************************************************************
 *
 *   Copyright (C) 2011 by Andrew Jameson and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SKLimits.h"
#include "Functor.h"

#include <iostream> 

using namespace std;

dsp::SKLimits::SKLimits (unsigned _M, unsigned _std_devs)
{
  M = _M;
  std_devs = _std_devs;
  lower_threshold = 0;
  upper_threshold = 0;
  verbose = 0;
}

dsp::SKLimits::~SKLimits ()
{

}

int dsp::SKLimits::calc_limits()
{

  if ((M == 0) || (std_devs == 0))
  {
    cerr << "SKLimits::calc_limits invalid inputs, M=" << M << " std_devs=" << std_devs << endl;
    lower_threshold = 0;
    upper_threshold = 0;
    return -1;
  }

  dsp::PearsonIV pIV(M);

  double percent_std_devs = erf((float) std_devs / sqrt(2));
  double target = (1 - percent_std_devs) / 2.0;
  double one_std_dev = sqrt(4.0 / (double) M);
  double factor = one_std_dev * std_devs;
  double x_guess = 0;

  if (verbose)
  {
    cerr << "SKLimits::calc_limits M=" << M << " std_devs=" << std_devs << " percent_std_devs=" << percent_std_devs << endl;
    cerr << "SKLimits::calc_limits 1 std_dev=" << one_std_dev << " factor=" << factor << endl;
    cerr << "SKLimits::calc_limits target=" << target << endl;
  }

  NewtonRaphson invert;
  invert.upper_limit = 10;
  invert.lower_limit = 1e-4;

  try
  {
    x_guess = 1 - factor;
    if (verbose)
      cerr << "SKLimits::calc_limits CF x_guess=" << x_guess << endl;
    Functor<double(double)> mycf ( &pIV, &dsp::PearsonIV::log_cf );
    Functor<double(double)> mydcf ( &pIV, &dsp::PearsonIV::dlog_cf );
    lower_threshold = invert(mycf, mydcf, log(target), x_guess);
  }
  catch (Error& error)
  {
    cerr << "SKLimits::calc_limits NewtonRaphson on CF failed" << endl;
    lower_threshold = x_guess;
  }

  try
  {
    x_guess = 1 + factor;
    if (verbose)
      cerr << "SKLimits::calc_limits CCF x_guess=" << x_guess << endl;
    Functor<double(double)> myccf ( &pIV, &PearsonIV::log_ccf );
    Functor<double(double)> mydccf ( &pIV, &PearsonIV::dlog_ccf );
    upper_threshold = invert(myccf, mydccf, log(target), x_guess);
  }
  catch (Error& error)
  {
    cerr << "SKLimits::calc_limits NewtonRaphson on CCF failed" << endl;
    upper_threshold = x_guess;
  }
  
  if (verbose)
    cerr << "SKLimits::calc_limits [" << lower_threshold << " - " << upper_threshold << "]" << endl;

  return 0;
}

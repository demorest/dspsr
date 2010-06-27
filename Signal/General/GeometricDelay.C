/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten et al
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/GeometricDelay.h"
#include "dsp/Observation.h"

#include <complex>

using namespace std;

dsp::GeometricDelay::GeometricDelay ()
{
  // HERE I would initialize any extra variables
}

#define SQR(x) (x*x)

bool dsp::GeometricDelay::match (const Observation* obs)
{
  /*
    HERE I would compute the geometric delay to apply as a number of samples
    using attributes of the Observation class.  These include MJD, R.A., Dec.

    The delays should be stored for later use by get_delay
  */  

  delay[0] = 0;
  delay[1] = 4;

  unsigned npol  = 2;    // representing two telescope
  unsigned nchan = 1;    // a single, undivide band of baseband data
  unsigned ndim  = 2;    // complex-valued response function
  uint64_t ndat  = 1024; // HERE: required frequency resolution

  response.resize (npol, nchan, ndat, ndim);

  // constant index
  const unsigned ichan = 0;

  // ///////////////////////////////////////////////////////////////////////
  //
  // setup for telescope 0
  //
  float* datptr = response.get_datptr (ichan, 0 /*telecope*/);

  complex<float>* phasors = (complex<float>*) datptr;
  for (unsigned idat=0; idat<ndat; idat++)
    phasors[idat] = polar (double(1.0), -1.0*idat*delay[0]*M_PI/ndat /* HERE */);

  // ///////////////////////////////////////////////////////////////////////
  //
  // setup for telescope 1
  //
  datptr = response.get_datptr (ichan, 1 /*telecope*/);

  phasors = (complex<float>*) datptr;
  for (unsigned idat=0; idat<ndat; idat++)
    phasors[idat] = polar (double(1.0), -1.0*idat*delay[1]*M_PI/ndat /* HERE */);

  return true;
}

//! Return the geometric delay for the given polarization
int64_t dsp::GeometricDelay::get_delay (unsigned ichan, unsigned ipol)
{
  return delay[ipol];
}


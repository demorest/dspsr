/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten et al
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/GeometricDelay.h"
#include "dsp/Observation.h"

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
  return true;
}

//! Return the geometric delay for the given polarization
int64_t dsp::GeometricDelay::get_delay (unsigned ichan, unsigned ipol)
{
  return 0; // delays[ipol];
}


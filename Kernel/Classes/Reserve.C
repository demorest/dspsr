/***************************************************************************
 *
 *   Copyright (C) 2013 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Reserve.h"
#include "dsp/TimeSeries.h"
#include "Error.h"

#include <iostream>

using namespace std;

dsp::Reserve::Reserve ()
{
  reserved = 0;
  sanity_check = NULL;
}

//! Set the number of samples to be reserved for this process
void dsp::Reserve::reserve (const TimeSeries* target, uint64_t samples)
{
  if (sanity_check && target != sanity_check)
    throw Error (InvalidParam, "dsp::Reserve::reserve",
		 "target has changed");

  if (reserved < samples)
  {
    if (TimeSeries::verbose)
      cerr << "dsp::Reserve::reserve"
              " increasing reserve to " << samples << endl;

    target->change_reserve (samples-reserved);
    reserved = samples;
  }

  sanity_check = target;
}

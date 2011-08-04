/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Resize.h"

#include "dsp/TimeSeries.h"

using namespace std;

dsp::Resize::Resize () 
  : Transformation<TimeSeries,TimeSeries>("Resize",inplace)
{
  resize_samples = 0;
}

dsp::Resize::~Resize ()
{
}

void dsp::Resize::set_resize_samples (int64_t samples)
{
  resize_samples = samples;
}


//! Perform the transformation on the input time series
void dsp::Resize::transformation ()
{
  
  if (resize_samples)
  {
    const uint64_t ndat = input->get_ndat();
    int64_t new_ndat = ndat + resize_samples;

    if (new_ndat < 0)
      new_ndat = 0;

    if (verbose)
      cerr << "dsp::Resize::transformation ndat " << ndat << " -> " << new_ndat << endl;

    output->set_ndat (new_ndat);
  }
}

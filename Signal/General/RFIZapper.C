/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/RFIZapper.h"

#include "dsp/TimeSeries.h"
#include "dsp/IOManager.h"
#include "dsp/Unpacker.h"
#include "dsp/Input.h"

#include "median_smooth.h"

using namespace std;

dsp::RFIZapper::RFIZapper () 
  : Transformation<TimeSeries,TimeSeries>("RFIZapper",inplace)
{
  M = 200;
}

dsp::RFIZapper::~RFIZapper ()
{
}

//! Set the raw/original time series [pre filterbank]
void dsp::RFIZapper::set_original_input (TimeSeries * _original_input)
{
  original_input = _original_input;
}

//! Perform the transformation on the input time series
void dsp::RFIZapper::transformation ()
{

  if (verbose)
    cerr << "dsp::RFIZapper::transformation input ndat="
         << input->get_ndat() << " ndim=" << input->get_ndim()
         << endl;

  // this transformation is on the output of the real
  // FFT, not the "original" fft. Here we need to check/peform
  // the TFPFilterbank on the corresponding data to see what
  // channels / samples may need to be zapped.

  // perform TFPFilterbank on corresponding data

  // apply zapping based on TFPFilterbank output

}

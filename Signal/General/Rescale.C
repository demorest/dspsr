/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Rescale.h"

using namespace std;

dsp::Rescale::Rescale ()
  : Transformation<TimeSeries,TimeSeries> ("Rescale", anyplace)
{
}

/*!
  \pre input TimeSeries must contain detected data
*/
void dsp::Rescale::transformation ()
{
  if (verbose)
    cerr << "dsp::Rescale::transformation" << endl;

  const uint64   input_ndat  = input->get_ndat();
  const unsigned input_ndim  = input->get_ndim();
  const unsigned input_npol  = input->get_npol();
  const unsigned input_nchan = input->get_nchan();

  if (input_ndim != 1)
    throw Error (InvalidState, "dsp::Rescale::transformation",
		 "invalid ndim=%d", input_ndim);

  uint64 output_ndat = input_ndat;

  // prepare the output TimeSeries
  output->copy_configuration (input);

  if (output != input)
    output->resize (output_ndat);
  else
    output->set_ndat (output_ndat);

  if (!output_ndat)
    return;

  for (unsigned ipol=0; ipol < input_npol; ipol++) 
  {
    for (unsigned ichan=0; ichan < input_nchan; ichan++)
    {
      const float* in_data = input->get_datptr (ichan, ipol);

      double sum = 0.0;
      double sumsq = 0.0;

      for (uint64 idat=0; idat < input_ndat; idat++)
      {
	sum += in_data[idat];
	sumsq += in_data[idat] * in_data[idat];
      }

      double mean = sum / input_ndat;
      double variance = sumsq/input_ndat - mean*mean;
      float scale = 1.0/sqrt(variance);

      float* out_data = output->get_datptr (ichan, ipol);

      for (uint64 idat=0; idat < output_ndat; idat++)
	out_data[idat] = (in_data[idat] - mean) * scale;
    }
  }
}

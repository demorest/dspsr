/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/PScrunch.h"

using namespace std;

dsp::PScrunch::PScrunch ()
  : Transformation<TimeSeries,TimeSeries> ("PScrunch", anyplace)
{
}

/*!
  \pre input TimeSeries must contain detected data
*/
void dsp::PScrunch::transformation ()
{
  if (verbose)
    cerr << "dsp::PScrunch::transformation" << endl;

  const uint64   input_ndat  = input->get_ndat();
  const unsigned input_ndim  = input->get_ndim();
  const unsigned input_npol  = input->get_npol();
  const unsigned input_nchan = input->get_nchan();

  if (input_ndim != 1)
    throw Error (InvalidState, "dsp::PScrunch::transformation",
		 "invalid ndim=%d", input_ndim);

  if (input_npol == 1)
    throw Error (InvalidState, "dsp::PScrunch::transformation",
		 "invalid npol=%d", input_npol);

  uint64 output_ndat = input_ndat;

  // prepare the output TimeSeries
  output->copy_configuration (input);
  output->set_npol (1);

  if (output != input)
    output->resize (output_ndat);
  else
    output->set_ndat (output_ndat);

  if (!output_ndat)
    return;

  for (unsigned ichan=0; ichan < input_nchan; ichan++)
  {
    const float* in_p0 = input->get_datptr (ichan, 0);
    const float* in_p1 = input->get_datptr (ichan, 1);

    float* out_data = output->get_datptr (ichan, 0);

    for (uint64 idat=0; idat < output_ndat; idat++)
      out_data[idat] = (in_p0[idat] + in_p1[idat]) * 0.5;
  }
}

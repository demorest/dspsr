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

  const uint64_t input_ndat  = input->get_ndat();
  const unsigned input_ndim  = input->get_ndim();
  const unsigned input_npol  = input->get_npol();
  const unsigned input_nchan = input->get_nchan();

  if (input_ndim != 1)
    throw Error (InvalidState, "dsp::PScrunch::transformation",
		 "invalid ndim=%d", input_ndim);

  if (input_npol == 1)
    throw Error (InvalidState, "dsp::PScrunch::transformation",
		 "invalid npol=%d", input_npol);

  uint64_t output_ndat = input_ndat;

  // prepare the output TimeSeries
  output->copy_configuration (input);

  if (output != input)
  {
    output->set_npol (1);
    output->resize (output_ndat);
  }

  if (output_ndat)
  {
    float scale = 1.0 / sqrt(2.0);
    
    switch (input->get_order())
    {
    case TimeSeries::OrderFPT:
    {
      // cerr << "dsp::PScrunch::transformation FPT order" << endl;

      for (unsigned ichan=0; ichan < input_nchan; ichan++)
      {
	const float* in_p0 = input->get_datptr (ichan, 0);
	const float* in_p1 = input->get_datptr (ichan, 1);
	
	float* out_data = output->get_datptr (ichan, 0);
	
	for (uint64_t idat=0; idat < output_ndat; idat++)
	  out_data[idat] = (in_p0[idat] + in_p1[idat]) * scale;
      }
      break;
    }
    case TimeSeries::OrderTFP:
    {
      // cerr << "dsp::PScrunch::transformation TFP order" << endl;

      int in,out;
      in=out=0;
      float* out_data = output->get_dattfp();
      const float* in_data = input->get_dattfp();
      for (uint64_t idat=0; idat < output_ndat; idat++)
      {
	for (unsigned ichan=0; ichan < input_nchan; ichan++)
        {
	  out_data[out] = (in_data[in] + in_data[in+1])*scale;
	  in+=input_npol;
	  out++;
	}
      }
      break;
    }
    default:
      throw Error (InvalidState, "dsp::PScrunch::operate",
		   "Can only handle data ordered TFP or FPT");
    }
  }

  // cerr << "ndat=" << output_ndat << endl;

  if (output == input)
    output->reshape (1,1);

  output->set_state (Signal::Intensity);
}


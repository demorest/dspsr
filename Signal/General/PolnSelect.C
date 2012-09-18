/***************************************************************************
 *
 *   Copyright (C) 2012 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/PolnSelect.h"

using namespace std;

dsp::PolnSelect::PolnSelect ()
  : Transformation<TimeSeries,TimeSeries> ("PolnSelect", anyplace)
{
  ipol_keep = 0;
}

/*!
  \pre input TimeSeries can contain detected or non-detected data
*/
void dsp::PolnSelect::transformation ()
{
  if (verbose)
    cerr << "dsp::PolnSelect::transformation" << endl;

  const uint64_t input_ndat  = input->get_ndat();
  const unsigned input_ndim  = input->get_ndim();
  const unsigned input_npol  = input->get_npol();
  const unsigned input_nchan = input->get_nchan();

  if ( (ipol_keep >= input_npol) || (ipol_keep < 0) )
    throw Error (InvalidParam, "dsp::PolnSelect::transformation",
		 "selected ipol=%d > input_npol=%d", ipol_keep, input_npol);

  uint64_t output_ndat = input_ndat;

  // prepare the output TimeSeries
  output->copy_configuration (input);

  if (output != input)
  {
    output->set_npol (1);
    output->set_ndim (input_ndim);
    output->resize (output_ndat);
  }

  if (output_ndat)
  {
    
    switch (input->get_order())
    {
    case TimeSeries::OrderFPT:
    {
      for (unsigned ichan=0; ichan < input_nchan; ichan++)
      {
        const float* in = input->get_datptr (ichan, ipol_keep);
	
        float* out_data = output->get_datptr (ichan, 0);
	
        for (uint64_t idat=0; idat < output_ndat*input_ndim; idat++)
          out_data[idat] = in[idat];
      }
      break;
    }
    case TimeSeries::OrderTFP:
    {
      int in,out;
      in=out=0;
      float* out_data = output->get_dattfp();
      const float* in_data = input->get_dattfp();
      for (uint64_t idat=0; idat < output_ndat; idat++)
      {
	for (unsigned ichan=0; ichan < input_nchan; ichan++)
        {
          for (unsigned idim=0; idim < input_ndim; idim++) 
            out_data[out+idim] = in_data[in+ipol_keep*input_ndim+idim];
	  in+=input_npol*input_ndim;
	  out+=input_ndim;
	}
      }
      break;
    }
    default:
      throw Error (InvalidState, "dsp::PolnSelect::transformation",
		   "Can only handle data ordered TFP or FPT");
    }
  }

  if (output == input)
    output->reshape (1,input_ndim);

  // Figure out what to call the output
  Signal::State in_state = input->get_state();
  if (in_state == Signal::Analytic || in_state == Signal::Nyquist
      || in_state == Signal::Intensity)
    output->set_state (in_state);
  else if (in_state == Signal::Stokes)
  {
    if (ipol_keep==0)
      output->set_state (Signal::Intensity);
    else
      output->set_state (Signal::Other);
  }
  else if (in_state == Signal::Coherence || in_state == Signal::PPQQ)
  {
    if (ipol_keep==0)
      output->set_state (Signal::PP_State);
    else if (ipol_keep==1)
      output->set_state (Signal::QQ_State);
    else
      output->set_state (Signal::Other);
  }
  else
    output->set_state (Signal::Other);
}


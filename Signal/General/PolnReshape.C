
/***************************************************************************
 *
 *   Copyright (C) 2015 by Matthew Kerr
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/PolnReshape.h"

using namespace std;

dsp::PolnReshape::PolnReshape ()
  : Transformation<TimeSeries,TimeSeries> ("PolnReshape", outofplace)
{
  state = Signal::Other;
}

void dsp::PolnReshape::npol4_ndim1()
{
  if (verbose)
    cerr << "dsp::PolnReshape::npol4_ndim1" << endl;
  const uint64_t ndat  = input->get_ndat();
  const unsigned nchan = input->get_nchan();

  output->set_npol(4);
  output->set_ndim(1);
  output->resize(ndat);

  switch (input->get_order())
  {
  case TimeSeries::OrderFPT:
  {
    for (unsigned ichan=0; ichan < nchan; ichan++)
    {
      for (unsigned ipol=0; ipol < 2; ipol++)
      {
        for (unsigned opol=0; opol < 2; opol++)
        {
          // offset by opol to get correct "dimension" member 
          const float* in_data = input->get_datptr (ichan, ipol) + opol;
          float* out_data = output->get_datptr (ichan, ipol*2 + opol);
          for (uint64_t idat=0; idat < ndat; idat++)
          {
            out_data[idat] = in_data[2*idat];
          }
        }
      }
    }
    break;
  }
  default :
    throw Error (InvalidState, "dsp::PolnReshape::npol4_ndim1",
     "Only FPT order implemented.");
  }
}

void dsp::PolnReshape::npol2_ndim1()
{
    throw Error (InvalidParam, "dsp::PolnReshape::npol2_ndim1",
		 "not implemented");
}

void dsp::PolnReshape::npol1_ndim1()
{
  if (verbose)
    cerr << "dsp::PolnReshape::npol1_ndim1" << endl;
  const uint64_t ndat  = input->get_ndat();
  const unsigned nchan = input->get_nchan();

  output->set_npol(1);
  output->set_ndim(1);
  output->resize(ndat);

  switch (input->get_order())
  {
  case TimeSeries::OrderFPT:
  {
    for (unsigned ichan=0; ichan < nchan; ichan++)
    {
      float* out_data = output->get_datptr (ichan, 0);
      const float* in_data = input->get_datptr (ichan, 0);
      // PP and QQ are stored in first input pol
      for (uint64_t idat=0; idat < ndat; idat++)
      {
        out_data[idat] = in_data[2*idat] + in_data[2*idat+1];
      }
    }
            
    break;
  }
  default :
    throw Error (InvalidState, "dsp::PolnReshape::npol1_ndim1",
     "Only FPT order implemented.");
  }
}

/*!
  \pre input TimeSeries must contain detected data with ndim=2, npol=2
*/
void dsp::PolnReshape::transformation ()
{
  if (verbose)
    cerr << "dsp::PolnReshape::transformation" 
         << " input state=" << tostring(input->get_state())
         << " output state=" << tostring(state) << endl;

  const uint64_t input_ndat  = input->get_ndat();
  const unsigned input_ndim  = input->get_ndim();
  const unsigned input_npol  = input->get_npol();

  if (!(input_npol==2 && input_ndim==2))
    throw Error (InvalidParam, "dsp::PolnReshape::transformation",
		 "npol=%d != 2 or ndim=%d != 2", input_npol, input_ndim);

  // prepare the output TimeSeries
  output->copy_configuration (input);

  if (output == input)
    throw Error (InvalidParam, "dsp::PolnReshape::transformation",
		 "does not currently support in-place transformation");

  if (state == Signal::Stokes || state == Signal::Coherence)
    npol4_ndim1();
  else if (state == Signal::PPQQ)
    npol2_ndim1();
  else if (state == Signal::Intensity)
    npol1_ndim1();
  else
    throw Error (InvalidParam, "dsp::PolnReshape::transformation",
		 "did not recognize input state");

  output->set_input_sample ( input->get_input_sample() );
}


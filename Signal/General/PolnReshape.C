
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

// reshape from 2pol, 2dim to 4pol, 1dim
void dsp::PolnReshape::p2d2_p4d1()
{
  if (verbose)
    cerr << "dsp::PolnReshape::p2d2_p4d1" << endl;
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
    throw Error (InvalidState, "dsp::PolnReshape::p2d2_p4d1",
     "Only FPT order implemented.");
  }
}

void dsp::PolnReshape::p2d2_p2d1()
{
    throw Error (InvalidParam, "dsp::PolnReshape::p2d2_p2d1",
		 "not implemented");
}

void dsp::PolnReshape::p2d2_p1d1()
{
  if (verbose)
    cerr << "dsp::PolnReshape::p2d2_p1d1" << endl;
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
    throw Error (InvalidState, "dsp::PolnReshape::p2d2_p1d1",
     "Only FPT order implemented.");
  }
}

void dsp::PolnReshape::p1d1_p1d1()
{
  if (verbose)
    cerr << "dsp::PolnReshape::p1d1_p1d1" << endl;
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
      for (uint64_t idat=0; idat < ndat; idat++)
      {
        out_data[idat] = in_data[idat];
      }
    }
           
    break;
  }  
  default :
    throw Error (InvalidState, "dsp::PolnReshape::p1d1_p1d1",
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

  if ((input_npol==1) && (input_ndim==1) && (state == Signal::Intensity))
  {
    if (verbose)
      cerr << "dsp::PolnReshape::transformation supported p1d1" << endl;
  }
  else if ((input_npol==2 && input_ndim==2) && 
           ((state == Signal::Stokes) || (state == Signal::Coherence) || 
            (state == Signal::PPQQ) || Signal::Intensity))
  {
    if (verbose)
      cerr << "dsp::PolnReshape::transformation supported p2d2 transformation" << endl;
  }
  else
    throw Error (InvalidParam, "dsp::PolnReshape::transformation",
		 "npol=%d, ndim=%d which is unsupported", input_npol, input_ndim);

  // prepare the output TimeSeries
  output->copy_configuration (input);

  if (output == input)
    throw Error (InvalidParam, "dsp::PolnReshape::transformation",
		 "does not currently support in-place transformation");

  if (input_npol==1 && input_ndim==1 && state == Signal::Intensity)
    p1d1_p1d1();
  else if (state == Signal::Stokes || state == Signal::Coherence)
    p2d2_p4d1();
  else if (state == Signal::PPQQ)
    p2d2_p2d1();
  else if (state == Signal::Intensity)
    p2d2_p1d1();
  else
    throw Error (InvalidParam, "dsp::PolnReshape::transformation",
		 "did not recognize input state");

  output->set_input_sample ( input->get_input_sample() );
}


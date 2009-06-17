/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SampleDelay.h"
#include "dsp/SampleDelayFunction.h"
#include "dsp/InputBuffering.h"

#include <assert.h>

using namespace std;

dsp::SampleDelay::SampleDelay ()
  : Transformation<TimeSeries,TimeSeries> ("SampleDelay", anyplace)
{
  zero_delay = 0;
  total_delay = 0;
  built = false;

  set_buffering_policy (new InputBuffering (this));
}

uint64_t dsp::SampleDelay::get_total_delay () const
{
  if (!built)
    const_cast<SampleDelay*>(this)->build();
  return total_delay;
}

int64_t dsp::SampleDelay::get_zero_delay () const
{
  if (!built)
    const_cast<SampleDelay*>(this)->build();
  return zero_delay;
}

//! Set the delay function
void dsp::SampleDelay::set_function (SampleDelayFunction* f)
{
  if (function && f == function.get())
    return;

  function = f;
  built = false;
}

void dsp::SampleDelay::build ()
{
  if (verbose)
    cerr << "dsp::SampleDelay::build" << endl;

  unsigned input_npol  = input->get_npol();
  unsigned input_nchan = input->get_nchan();

  zero_delay = function->get_delay (0, 0);

  for (unsigned ipol=0; ipol < input_npol; ipol++)
    for (unsigned ichan=0; ichan < input_nchan; ichan++)
      if (function->get_delay (ichan, ipol) > zero_delay)
	zero_delay = function->get_delay (ichan, ipol);

  if (verbose)
    cerr << "dsp::SampleDelay::build zero delay = " << zero_delay 
	 << " samples" << endl;

  total_delay = 0;

  for (unsigned ipol=0; ipol < input_npol; ipol++)
    for (unsigned ichan=0; ichan < input_nchan; ichan++) {
      uint64_t relative_delay = zero_delay - function->get_delay(ichan, ipol);
      // cerr << "relative_delay=" << relative_delay << endl;
      if (relative_delay > total_delay)
	total_delay = relative_delay;
    }

  if (verbose)
    cerr << "dsp::SampleDelay::build total delay = " << total_delay 
	 << " samples" << endl;

  built = true;
}

void dsp::SampleDelay::prepare ()
{
  if (function->match(input) || !built)
    build ();

  if (!has_buffering_policy())
    return;

  if (verbose)
    cerr << "dsp::SampleDelay::prepare reserve=" << total_delay << endl;

  get_buffering_policy()->set_minimum_samples (total_delay);
}

/*!
  \pre input TimeSeries must contain complex (Analytic) data
*/
void dsp::SampleDelay::transformation ()
{
  if (verbose)
    cerr << "dsp::SampleDelay::transformation" << endl;

  const uint64_t   input_ndat  = input->get_ndat();
  const unsigned input_ndim  = input->get_ndim();
  const unsigned input_npol  = input->get_npol();
  const unsigned input_nchan = input->get_nchan();

  uint64_t output_ndat = 0;

  if (input_ndat < total_delay)
  {
    if (verbose)
      cerr << "dsp::SampleDelay::transformation insufficient data\n"
        "  input ndat=" << input_ndat << " total delay=" << total_delay  << endl;
  }
  else
    output_ndat = input_ndat - total_delay;

  get_buffering_policy()->set_next_start (output_ndat);

  // prepare the output TimeSeries
  output->copy_configuration (input);

  if (output != input)
    output->resize (output_ndat);
  else
    output->set_ndat (output_ndat);

  // zero_delay
  output->change_start_time (zero_delay);

  output->check_sanity ();

  if (!output_ndat)
    return;

  uint64_t output_nfloat = output_ndat * input_ndim;

  for (unsigned ipol=0; ipol < input_npol; ipol++) {

    for (unsigned ichan=0; ichan < input_nchan; ichan++) {

      const float* in_data = input->get_datptr (ichan, ipol);
      int64_t relative_delay = zero_delay - function->get_delay(ichan, ipol);
      assert (relative_delay >= 0);

#ifdef _DEBUG
      cerr << "ipol=" << ipol << " ichan=" << ichan 
	   << " delay=" << relative_delay << endl;
#endif

      in_data += relative_delay * input_ndim;

      float* out_data = output->get_datptr (ichan, ipol);

      for (uint64_t idat=0; idat < output_nfloat; idat++)
	out_data[idat] = in_data[idat];

    }

  }

  function -> mark (output);
}

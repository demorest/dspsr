/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Rescale.h"
#include "dsp/InputBuffering.h"

#include <assert.h>

using namespace std;

dsp::Rescale::Rescale ()
  : Transformation<TimeSeries,TimeSeries> ("Rescale", anyplace)
{
  nsample = isample = 0;
  interval_seconds = 0.0;
  interval_samples = 0;
  exact = false;
  constant_offset_scale = false;
  output_time_total = false;
  output_after_interval = false;
  do_decay=false;
  decay_constant=1e4;
}

dsp::Rescale::~Rescale ()
{
  // cerr << "dsp::Rescale::~Rescale isample=" << isample << endl;

  if (isample)
  {
    compute_various ();
    update (this);
  }
}

void dsp::Rescale::set_output_after_interval (bool flag)
{
  output_after_interval = flag;
}

void dsp::Rescale::set_output_time_total (bool flag)
{
  output_time_total = flag;
}

void dsp::Rescale::set_constant (bool value)
{
  constant_offset_scale = value;
}


void dsp::Rescale::set_decay (float _decay_constant){
	if (_decay_constant > 0)
		do_decay=true;
	decay_constant=_decay_constant;
}

//! Set the rescaling interval in seconds
void dsp::Rescale::set_interval_seconds (double seconds)
{
  interval_seconds = seconds;
}

//! Set the rescaling interval in samples
void dsp::Rescale::set_interval_samples (uint64_t samples)
{
  interval_samples = samples;
}

//! Set the rescaling interval in samples
void dsp::Rescale::set_exact (bool value)
{
  exact = value;
  if (!has_buffering_policy())
    set_buffering_policy( new InputBuffering (this) );
  if (exact && !interval_samples)
      throw Error(InvalidState, "dsp::Rescale::set_exact", 
          "interval_sample == 0 (must be set)");
  get_buffering_policy()->set_minimum_samples (interval_samples);
}

template<typename T>
void zero (vector<T>& data)
{
  const unsigned n = data.size();
  for (unsigned i=0; i<n; i++)
    data[i]=0;
}

void dsp::Rescale::init ()
{
  const unsigned input_npol  = input->get_npol();
  const unsigned input_nchan = input->get_nchan();

  if (verbose)
    cerr << "dsp::Rescale::init npol=" << input_npol 
	 << " nchan=" << input_nchan 
	 << " ndat=" << input->get_ndat() << endl;

  if (interval_samples)
    nsample = interval_samples;
  else if (interval_seconds)
    nsample = uint64_t( interval_seconds * input->get_rate() );
  else
    nsample = input->get_ndat ();

  if (verbose)
    cerr << "dsp::Rescale::init interval samples = " << nsample << endl;

  if (!nsample)
  {
    Error error (InvalidState, "dsp::Rescale::init", "nsample == 0");
    error << " (interval samples=" << interval_samples
	  << " seconds=" << interval_seconds << ")";
    throw error;
  }
 
  isample = 0;

  if (output_time_total)
    time_total.resize (input_npol);
  freq_total.resize (input_npol);
  freq_totalsq.resize (input_npol);

  scale.resize (input_npol);
  offset.resize (input_npol);

  if(do_decay)
	  decay_offset.resize (input_npol);

  for (unsigned ipol=0; ipol < input_npol; ipol++)
  {
    if (output_time_total)
    {
      time_total[ipol].resize (nsample);
      zero (time_total[ipol]);
    }
  
    freq_total[ipol].resize (input_nchan);
    zero (freq_total[ipol]);

    freq_totalsq[ipol].resize (input_nchan);
    zero (freq_total[ipol]);

    scale[ipol].resize (input_nchan);
    offset[ipol].resize (input_nchan);

    if (do_decay){
	    decay_offset[ipol].resize(input_nchan);
	    zero (decay_offset[ipol]);
    }
  }
}

void dsp::Rescale::prepare ()
{
}

/*!
  \pre input TimeSeries must contain detected data
*/
void dsp::Rescale::transformation ()
{
  if (verbose)
    cerr << "dsp::Rescale::transformation" << endl;

  // if requested a minimum number of samples, let input buffering handle it
  if (exact && (input->get_ndat() < interval_samples))
  {
    if (verbose)
      cerr << "dsp::Rescale::transformation waiting for additional samples"            << endl;
    get_buffering_policy()->set_next_start ( 0 );
    output->set_ndat (0);
    return;
  }

  bool first_call = nsample == 0;

  if (first_call)
    init ();

  const uint64_t input_ndat  = input->get_ndat();
  const unsigned input_ndim  = input->get_ndim();
  const unsigned input_npol  = input->get_npol();
  const unsigned input_nchan = input->get_nchan();

  if (verbose)
    cerr << "dsp::Rescale::transformation input_ndat=" << input_ndat 
	 << " nsample=" << nsample << endl;

  if (input_ndim != 1)
    throw Error (InvalidState, "dsp::Rescale::transformation",
		 "invalid ndim=%d", input_ndim);

  uint64_t output_ndat = input_ndat;

  // prepare the output TimeSeries
  output->copy_configuration (input);

  // Since we will be rescaling data, remove any pre-set scale
  // factors (for example Filterbank/FFT normalizations).
  output->set_scale(1.0);

  if (output != input)
    output->resize (output_ndat);
  else
    output->set_ndat (output_ndat);

  if (!output_ndat)
    return;

  uint64_t start_dat = 0;
  uint64_t end_dat = input_ndat;

  do
  {
    end_dat = exact ? interval_samples : input_ndat;

    uint64_t interval_end_dat = start_dat + nsample - isample;
    if (interval_end_dat < end_dat)
      end_dat = interval_end_dat;

    uint64_t samp_dat = isample;

    if (verbose)
      cerr << "dsp::Rescale::transformation end_dat=" << end_dat
           << " interval_end_dat=" << interval_end_dat
           << " isample=" << isample << endl;

    switch (input->get_order()) {
    case TimeSeries::OrderTFP:
	  {
      const float* in_data = input->get_dattfp();
      in_data += start_dat * input_nchan*input_npol;
      for (unsigned idat=start_dat; idat < end_dat; idat++)
      {
        for (unsigned ichan=0; ichan < input_nchan; ichan++)
        {
          for (unsigned ipol=0; ipol < input_npol; ipol++)
          {
            freq_total[ipol][ichan]  += (*in_data);
            freq_totalsq[ipol][ichan]  += (*in_data)*(*in_data);

            if (output_time_total)
              time_total[ipol][samp_dat] += (*in_data);
            in_data++;

	        }
	      }
	      samp_dat++;
	    }
	    break;
	  }
    case TimeSeries::OrderFPT:
	  {
      for (unsigned ipol=0; ipol < input_npol; ipol++) 
      {
        for (unsigned ichan=0; ichan < input_nchan; ichan++)
        {
          const float* in_data = input->get_datptr (ichan, ipol);

          samp_dat = isample;

          double sum = 0.0;
          double sumsq = 0.0;

          for (unsigned idat=start_dat; idat < end_dat; idat++)
          {
            sum += in_data[idat];
            sumsq += in_data[idat] * in_data[idat];

            if (output_time_total)
              time_total[ipol][samp_dat] += in_data[idat];

            samp_dat++;
          }

          freq_total[ipol][ichan] += sum;
          freq_totalsq[ipol][ichan] += sumsq;
        }
      }
	    break;
	  }
    default:
	    throw Error (InvalidState, "dsp::Rescale::operate",
		      "Requires data in TFP or FPT order");

    }
    isample = samp_dat;

   if (samp_dat == nsample || first_call)
   {
    if (verbose)
      cerr << "dsp::Rescale::transformation rescale"
           << " nsample=" << nsample
           << " isample=" << isample 
           << " first_call=" << first_call << endl;

    if (first_call)
      update_epoch = input->get_start_time();

    compute_various (first_call);
    update (this);

    update_epoch += isample / input->get_rate();
    isample = 0;
    first_call = false;

    for (unsigned ipol=0; ipol < input_npol; ipol++)
    {
      zero (freq_total[ipol]);
      zero (freq_totalsq[ipol]);
      if (output_time_total)
        zero (time_total[ipol]);
    }
  }

  switch(input->get_order()) {

  case TimeSeries::OrderTFP:
	{
	  float tmp;
	  const float* in_data = input->get_dattfp();
	  float* out_data = output->get_dattfp();
	  in_data += start_dat * input_nchan*input_npol;
	  out_data += start_dat * input_nchan*input_npol;
	  for (unsigned idat=start_dat; idat < end_dat; idat++)
    {
	    for (unsigned ichan=0; ichan < input_nchan; ichan++)
      {
	      for (unsigned ipol=0; ipol < input_npol; ipol++)
        {
          if (do_decay)
          {
            tmp= ((*in_data) + offset[ipol][ichan]) * scale[ipol][ichan];
            decay_offset[ipol][ichan] = (tmp + decay_offset[ipol][ichan]*decay_constant) / (1.0+ decay_constant);
            (*out_data) = tmp - decay_offset[ipol][ichan];
          } 
          else {
            (*out_data) = ((*in_data) + offset[ipol][ichan]) * scale[ipol][ichan];
          }
          in_data++;
          out_data++;
	      }
	    }
	  }
	  break;
	}

  case TimeSeries::OrderFPT:
	{
	  for (unsigned ipol=0; ipol < input_npol; ipol++) 
	  {
	    for (unsigned ichan=0; ichan < input_nchan; ichan++)
		  {
        const float* in_data = input->get_datptr (ichan, ipol);
        float* out_data = output->get_datptr (ichan, ipol);

        float the_offset = offset[ipol][ichan];
        float the_scale = scale[ipol][ichan];
        for (uint64_t idat=start_dat; idat < end_dat; idat++)
          out_data[idat] = (in_data[idat] + the_offset) * the_scale;
		  }
	  }
	  break;
	}
  default:
	  throw Error (InvalidState, "dsp::Rescale::operate",
		     "Requires data in TFP or FPT order");
  }

  start_dat = end_dat;

  if (verbose)
	  cerr << "end_dat=" << end_dat << " input_ndat=" << input_ndat << endl;

  if (exact) break;
  }
  while (end_dat < input_ndat);

  if (exact)
    get_buffering_policy()->set_next_start ( interval_samples );

  if (verbose)
    cerr << "dsp::Rescale::transformation exit" << endl;
}

void dsp::Rescale::compute_various (bool first_call)
{
  // cerr << "dsp::Rescale::compute_various isample=" << isample << endl;

  const unsigned input_npol  = input->get_npol();
  const unsigned input_nchan = input->get_nchan();

  for (unsigned ipol=0; ipol < input_npol; ipol++)
  {
    for (unsigned ichan=0; ichan < input_nchan; ichan++)
    {
      double mean = freq_total[ipol][ichan] / isample;
      double meansq = freq_totalsq[ipol][ichan] / isample;
      double variance = meansq - mean*mean;

      freq_total[ipol][ichan] = mean;
      freq_totalsq[ipol][ichan] = variance;

      if (!constant_offset_scale || first_call)
      {
        offset[ipol][ichan] = -mean;
        if (variance == 0.0)
          scale[ipol][ichan] = 1.0;
        else
          scale[ipol][ichan] = 1.0 / sqrt(variance);
      }
    }
  }
}

//! Get the epoch of the last scale/offset update
MJD dsp::Rescale::get_update_epoch () const
{
  return update_epoch;
}

//! Get the mean bandpass for the given polarization
const float* dsp::Rescale::get_offset (unsigned ipol) const
{
  assert (ipol < offset.size());
  return &(offset[ipol][0]);
}

//! Get the rms bandpass for the given polarization
const float* dsp::Rescale::get_scale (unsigned ipol) const
{
  assert (ipol < scale.size());
  return &(scale[ipol][0]);
}

//! Get the mean bandpass for the given polarization
const double* dsp::Rescale::get_mean (unsigned ipol) const
{
  assert (ipol < freq_total.size());
  return &(freq_total[ipol][0]);
}

//! Get the scale bandpass for the given polarization
const double* dsp::Rescale::get_variance (unsigned ipol) const
{
  assert (ipol < freq_totalsq.size());
  return &(freq_totalsq[ipol][0]);
}

//! Get the number of samples between updates
uint64_t dsp::Rescale::get_nsample () const
{
  return nsample;
}

//! Get the total power time series for the given polarization
const float* dsp::Rescale::get_time (unsigned ipol) const
{
  assert (ipol < time_total.size());
  return &(time_total[ipol][0]);
}


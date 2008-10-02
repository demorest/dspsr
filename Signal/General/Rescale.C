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
  nsample = isample = 0;
  interval_seconds = 0.0;
  interval_samples = 0;
}

//! Set the rescaling interval in seconds
void dsp::Rescale::set_interval_seconds (double seconds)
{
  interval_seconds = seconds;
}

//! Set the rescaling interval in samples
void dsp::Rescale::set_interval_samples (uint64 samples)
{
  interval_samples = samples;
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

  scale.resize (input_npol);
  for (unsigned ipol=0; ipol < input_npol; ipol++)
    scale[ipol].resize (input_nchan);

  if ( ! (interval_seconds || interval_samples) )
  {
    nsample = isample = 0;
    return;
  }

  if (interval_samples)
    nsample = interval_samples;
  else
    nsample = interval_seconds / input->get_rate();

  cerr << "dsp::Rescale::init interval samples = " << nsample << endl;

  isample = 0;

  time_total.resize (input_npol);
  freq_total.resize (input_npol);
  freq_totalsq.resize (input_npol);

  for (unsigned ipol=0; ipol < input_npol; ipol++)
  {
    time_total[ipol].resize (nsample);
    zero (time_total[ipol]);

    freq_total[ipol].resize (input_nchan);
    zero (freq_total[ipol]);

    freq_totalsq[ipol].resize (input_nchan);
    zero (freq_total[ipol]);
  }
}

/*!
  \pre input TimeSeries must contain detected data
*/
void dsp::Rescale::transformation ()
{
  if (verbose)
    cerr << "dsp::Rescale::transformation" << endl;

  bool first_call = scale.size () == 0;

  if (first_call)
    init ();

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

  uint64 start_dat = 0;
  uint64 end_dat = input_ndat;

  do
  {
    end_dat = input_ndat;

    if (nsample)
    {
      uint64 interval_end_dat = nsample - isample;
      if (interval_end_dat < end_dat)
	end_dat = interval_end_dat;
    }

    uint64 samp_dat = isample;

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

	  if (nsample)
	    time_total[ipol][samp_dat] += in_data[idat];

	  samp_dat++;
	}

	freq_total[ipol][ichan] += sum;
	freq_totalsq[ipol][ichan] += sumsq;
      }
    }

    isample = samp_dat;

    if (!nsample || samp_dat == nsample || first_call)
    {
      uint64 count = nsample;

      if (!nsample || first_call)
	count = input_ndat;
      
      first_call = false;

      for (unsigned ipol=0; ipol < input_npol; ipol++) 
      {
	for (unsigned ichan=0; ichan < input_nchan; ichan++)
	{
	  double mean = freq_total[ipol][ichan] / count;
	  double meansq = freq_totalsq[ipol][ichan] / count;
	  double variance = meansq - mean*mean;

	  freq_total[ipol][ichan] = mean;
	  freq_totalsq[ipol][ichan] = variance;

	  offset[ipol][ichan] = -mean;
	  scale[ipol][ichan] = 1.0 / sqrt(variance);
	}
      }

      update (this);

      for (unsigned ipol=0; ipol < input_npol; ipol++)
      {
	zero (freq_total[ipol]);
	zero (freq_totalsq[ipol]);
	zero (time_total[ipol]);
      }
    }

    for (unsigned ipol=0; ipol < input_npol; ipol++) 
    {
      for (unsigned ichan=0; ichan < input_nchan; ichan++)
      {
	const float* in_data = input->get_datptr (ichan, ipol);
	float* out_data = output->get_datptr (ichan, ipol);

	float the_offset = offset[ipol][ichan];
	float the_scale = scale[ipol][ichan];
	for (uint64 idat=start_dat; idat < end_dat; idat++)
	  out_data[idat] = (in_data[idat] + the_offset) * the_scale;
      }
    }

    start_dat = end_dat;
  }
  while (end_dat < input_ndat);
}


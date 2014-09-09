/***************************************************************************
 *
 *   Copyright (C) 2014 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/ConstantRescale.h"
#include "ThreadContext.h"

#include <assert.h>

using namespace std;

dsp::ConstantRescale::ConstantRescale ()
  : Transformation<TimeSeries,TimeSeries> ("ConstantRescale", anyplace)
{ 
  share = NULL;
}

dsp::ConstantRescale::~ConstantRescale ()
{
}

dsp::ConstantRescale::ScaleOffsetShare::ScaleOffsetShare ()
{
  context = NULL;
  computed = false;
}

dsp::ConstantRescale::ScaleOffsetShare::~ScaleOffsetShare ()
{
}

template<typename T>
void zero (vector<T>& data)
{
  const unsigned n = data.size();
  for (unsigned i=0; i<n; i++)
    data[i]=0;
}

void dsp::ConstantRescale::ScaleOffsetShare::compute (
    const dsp::TimeSeries* input)
{
  ThreadContext::Lock lock (context);

  // another thread computed levels first, so return
  if (computed) return;

  // Resize, init arrays

  const uint64_t input_ndat  = input->get_ndat();
  const unsigned input_ndim  = input->get_ndim();
  const unsigned input_npol  = input->get_npol();
  const unsigned input_nchan = input->get_nchan();

  vector< vector<double> > sum;
  vector< vector<double> > sumsq;

  scale.resize (input_npol);
  offset.resize (input_npol);
  sum.resize (input_npol);
  sumsq.resize (input_npol);
  for (unsigned ipol=0; ipol<input_npol; ipol++) 
  {
    scale[ipol].resize (input_nchan);
    offset[ipol].resize (input_nchan);

    sum[ipol].resize (input_nchan);
    sumsq[ipol].resize (input_nchan);

    zero(sum[ipol]);
    zero(sumsq[ipol]);
  }

  // Do the computation

  if (input->get_order() == TimeSeries::OrderTFP) 
  {
    const float* in_data = input->get_dattfp();
    for (unsigned idat=0; idat<input_ndat; idat++) {
      for (unsigned ichan=0; ichan<input_nchan; ichan++) {
        for (unsigned ipol=0; ipol<input_npol; ipol++) {
          sum[ipol][ichan] += (*in_data);
          sumsq[ipol][ichan] += (*in_data)*(*in_data);
          in_data++;
        }
      }
    }
  }
  
  else if (input->get_order() == TimeSeries::OrderFPT)
  {
    for (unsigned ipol=0; ipol<input_npol; ipol++) {
      for (unsigned ichan=0; ichan<input_nchan; ichan++) {
        const float *in_data = input->get_datptr(ichan, ipol);
        for (unsigned idat=0; idat<input_ndat; idat++) {
          sum[ipol][ichan] += (*in_data);
          sumsq[ipol][ichan] += (*in_data)*(*in_data);
          in_data++;
        }
      }
    }
  }

  else
    throw Error (InvalidState, "dsp::ConstantRescale::compute_levels",
        "Requires data in TFP or FPT order");

  for (unsigned ipol=0; ipol<input_npol; ipol++) {
    for (unsigned ichan=0; ichan<input_nchan; ichan++) {
      double mean = sum[ipol][ichan]/(double)input_ndat;
      double var = sumsq[ipol][ichan]/(double)input_ndat - mean*mean;

      offset[ipol][ichan] = -mean;

      if (var<=0.0) 
        scale[ipol][ichan] = 1.0;
      else
        scale[ipol][ichan] = 1.0/sqrt(var);

    }
  }

  computed = true;
}

void dsp::ConstantRescale::ScaleOffsetShare::get_scale_offset (
    const dsp::TimeSeries* input,
    vector< vector<float > >& _scale, 
    vector< vector<float > >& _offset)
{
  if (!computed) compute(input);
  _scale = scale;
  _offset = offset;
}

/*!
  \pre input TimeSeries must contain detected data
*/
void dsp::ConstantRescale::transformation ()
{
  if (verbose)
    cerr << "dsp::ConstantRescale::transformation" << endl;

  if (!share) share = new ScaleOffsetShare;

  const uint64_t input_ndat  = input->get_ndat();
  const unsigned input_ndim  = input->get_ndim();
  const unsigned input_npol  = input->get_npol();
  const unsigned input_nchan = input->get_nchan();

  if (input_ndim != 1)
    throw Error (InvalidState, "dsp::ConstantRescale::transformation",
		 "invalid ndim=%d", input_ndim);

  uint64_t output_ndat = input_ndat;

  // prepare the output TimeSeries
  output->copy_configuration (input);

  if (output != input)
    output->resize (output_ndat);
  else
    output->set_ndat (output_ndat);

  if (!output_ndat)
    return;

  uint64_t start_dat = 0;
  uint64_t end_dat = input_ndat;

  share->get_scale_offset(input, scale, offset);

  if (input->get_order() == TimeSeries::OrderTFP)
  {
    float tmp;
    const float* in_data = input->get_dattfp();
    float* out_data = output->get_dattfp();
    in_data += start_dat * input_nchan*input_npol;
    out_data += start_dat * input_nchan*input_npol;
    for (unsigned idat=start_dat; idat < end_dat; idat++) {
      for (unsigned ichan=0; ichan < input_nchan; ichan++) {
        for (unsigned ipol=0; ipol < input_npol; ipol++) {
          (*out_data) = ((*in_data) + offset[ipol][ichan]) 
            * scale[ipol][ichan];
          }
          in_data++;
          out_data++;
        }
      }
  }

  else if (input->get_order() == TimeSeries::OrderFPT)
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
  }

  else
    throw Error (InvalidState, "dsp::ConstantRescale::operate",
               "Requires data in TFP or FPT order");

  if (verbose)
    cerr << "dsp::ConstantRescale::transformation exit" << endl;
}


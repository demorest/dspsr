/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/ZapWeight.h"

#include "dsp/UniversalInputBuffering.h"

#include "dsp/TimeSeries.h"
#include "dsp/IOManager.h"
#include "dsp/Unpacker.h"
#include "dsp/Input.h"

#include <assert.h>

using namespace std;

dsp::ZapWeight::ZapWeight () 
  : Transformation<TimeSeries,TimeSeries>("ZapWeight",inplace)
{
  M = 0;
  sigma = 0;
  upper_thresh = 0;
  lower_thresh = 0;
  mega_upper_thresh = 0;
  mega_lower_thresh = 0;
  debugd = 1;
  ddfb_rate = 0;
  skfb_rate = 0;
  total_idats = 0;
  total_zaps = 0;

  set_buffering_policy (new_UniversalInputBuffering (this, &ZapWeight::get_skfb_input));
}

dsp::ZapWeight::~ZapWeight ()
{
}

void dsp::ZapWeight::set_engine (Engine * _engine)
{
  engine = _engine;
}

//! Set the SKFilterbank input
void dsp::ZapWeight::set_skfb_input (TimeSeries * _skfb_input)
{
  skfb_input = _skfb_input;
}

const dsp::TimeSeries* dsp::ZapWeight::get_skfb_input () const
{
  return skfb_input;
}

void dsp::ZapWeight::set_M (unsigned _M)
{
  M = _M;
  sigma = sqrtf(4.0 / (float) M);
}

void dsp::ZapWeight::set_thresholds (float factor)
{
  if (sigma)
  {
    upper_thresh = 1 + (factor * sigma);
    lower_thresh = 1 - (factor * sigma);
  }

  mega_upper_thresh = 1 + (10.0 * sigma);
  mega_lower_thresh = 1 - (10.0 * sigma);

  if (verbose)
    cerr << "dsp::ZapWeight::set_thresholds " << lower_thresh << " - " 
         << upper_thresh << endl;
}


//! Perform the transformation on the input time series
void dsp::ZapWeight::transformation ()
{

  if (engine) 
  {
    //if (verbose)
      cerr << "dsp::ZapWeight::transformation using Engine" << endl;
    engine->perform (input, output);

  }

  ddfb_rate = input->get_rate();
  skfb_rate = skfb_input->get_rate();

  if (verbose || debugd < 1)
    cerr << "dsp::ZapWeight::transformation input ndat="
         << input->get_ndat() << " ndim=" << input->get_ndim()
         << " nchan=" << input->get_nchan() << endl;

  const float * indat = input->get_datptr();
  const float * skfb = skfb_input->get_dattfp ();

  const unsigned ddfb_nchan = input->get_nchan();
  const uint64_t ddfb_ndat  = input->get_ndat();
  const uint64_t ddfb_npol  = input->get_npol();

  const unsigned skfb_nchan = skfb_input->get_nchan();
  const unsigned skfb_npol  = skfb_input->get_npol();
  int64_t skfb_ndat  = skfb_input->get_ndat();

  const uint64_t ddfb_input_sample = input->get_input_sample();
  const uint64_t skfb_input_sample = skfb_input->get_input_sample();

  const unsigned output_ndim = output->get_ndim();

  if (debugd < 1)
  {
    cerr << "dsp::ZapWeight::transformation DDFB: nchan=" << ddfb_nchan
         << " npol=" << ddfb_npol << " ndat=" << ddfb_ndat << endl;
    cerr << "dsp::ZapWeight::transformation SKFB: nchan=" << skfb_nchan
         << " npol=" << skfb_npol << " ndat=" << skfb_ndat << endl;

    cerr << "dsp::ZapWeight::transformation SKFB rate=" << skfb_input->get_rate() << endl;
    cerr << "dsp::ZapWeight::transformation DDFB rate=" << input->get_rate() << endl;

    cerr << "dsp::ZapWeight::transformation SKFB samples=" << skfb_input_sample << " -> " << (skfb_input_sample + skfb_ndat) << endl;
    cerr << "dsp::ZapWeight::transformation DDFB samples=" << ddfb_input_sample << " -> " << (ddfb_input_sample + ddfb_ndat) << endl;
    cerr << "dsp::ZapWeight::transformation output_ndim=" << output_ndim << endl;
  }

  float V_p0;
  float V_p1;

  // calculate the offset between the ddfb and skfb in ddfb samples
  MJD offset_mjd = input->get_start_time() - skfb_input->get_start_time();
  int64_t ddfb_offset_samples = (int64_t) (offset_mjd.in_seconds() * ddfb_rate);
  if (debugd < 1)
    cerr << "dsp::ZapWeight::transformation ddfb_offset_samples=" << ddfb_offset_samples << endl;

  int64_t weight_idat = ddfb_offset_samples % M;
  int64_t weight_offset = ddfb_offset_samples / M;

  if (debugd < 1)
    cerr << "dsp::ZapWeight::transformation weight_idat=" << weight_idat 
         << " weight_offset=" << weight_offset << endl;

  if (ddfb_offset_samples < 0)
    cerr << "dsp::ZapWeight::transformation ddfb_offset_samples < 0 !!!!" << endl;

  // indicate the output timeseries contains zeroed data
  output->set_zeroed_data (true);

  // WvS: weight_idat must be set before calling resize
  output->resize (output->get_ndat());
  //output->neutral_weights ();

  uint64_t skfb_idat = 0;

  unsigned zap_all_chan = 0;
  unsigned zap_chan_count = 0;
  uint64_t zap_count = 0;
  uint64_t output_index = 0;

  // if the DDFB input sample is more than 1 SKFB block, adjust SKFB ptr and ndat
  if (weight_offset) 
  {
    cerr << "dsp::ZapWeight::transformation weight_offest > 0!!!!!!!!!!!!!" << endl;
    skfb += weight_offset * skfb_nchan * skfb_npol;
    skfb_ndat -= weight_offset;
  }

  // the SKFB might extend past the end of the DDFB input data block
  uint64_t skfb_ndat_needed = ddfb_ndat / M;
  if (ddfb_ndat % M)
  {
    skfb_ndat_needed++;
  }

  // truncate ddfb if there 
  if (skfb_ndat_needed > skfb_ndat)
  {
    cerr << "dsp::ZapWeight::transformation WHOA WE NEED MORE SKFB THAN WE HAVE" << endl;
    output->set_ndat (skfb_ndat * M);
    cerr << "dsp::ZapWeight::transformation truncating DDFB ndat to " << (skfb_ndat * M) << endl;
  }

  if (debugd < 1)
    cerr << "dsp::ZapWeight::transformation ddfb_ndat/M=" << (ddfb_ndat / M) << " ddfb_ndat%M=" << (ddfb_ndat % M) << " skfb_ndat_needed=" << skfb_ndat_needed << endl;

  if (skfb_ndat_needed < skfb_ndat)
  {
    if (verbose)
      cerr << "CUT OUT EARLY limiting skfb_ndat to " << skfb_ndat_needed << ", was " << skfb_ndat << endl;
    skfb_ndat = skfb_ndat_needed;
  }

  // the number of samples ddfb samples are offset from ddfb->get_datptr()
  unsigned ddfb_offset = weight_idat;
  uint64_t ddfb_start_idat;
  uint64_t ddfb_nsamples;
  uint64_t ddfb_end_idat;

  for (uint64_t idat=0; idat < skfb_ndat; idat++)
  {
    ddfb_start_idat = idat * M;
    ddfb_nsamples   = M;
    ddfb_end_idat   = ddfb_start_idat + ddfb_nsamples;

    // the DDFB does not span this SKFB idat
    if (ddfb_start_idat >= ddfb_ndat)
    {
      cerr << "dsp::ZapWeight::transformation BREAK!!! idat=" << idat 
           << " ddfb_start_idat[" << ddfb_start_idat << "] > ddfb_ndat [" 
           << ddfb_ndat << "]" << endl;
      break;
    }

    // if the DDFB only partially spans this SKFB 
    if (ddfb_end_idat > ddfb_ndat)
    {
      if (verbose)
        cerr << "dsp::ZapWeight::transformation ddfb_end_idat [" 
             << ddfb_end_idat << "] > ddfb_ndat [" << ddfb_ndat 
              << "]" << endl;
      ddfb_nsamples -= (ddfb_end_idat - ddfb_ndat);
      ddfb_end_idat = ddfb_ndat;
    }
    else
    {
      if (ddfb_offset > ddfb_end_idat)
      {
        cerr << "dsp::ZapWeight::transformation ddfb_offset > ddfb_end_idat" << endl;
        break;
      }
      else
        ddfb_end_idat -= ddfb_offset;
    }

    // if we are on the first idat, 
    if (ddfb_offset > ddfb_start_idat)
      ddfb_nsamples -= ddfb_offset;
    else
      ddfb_start_idat -= ddfb_offset;

    zap_all_chan = 0;
    zap_chan_count = 0;

    // for each channel/pol in the SK FB
    for (unsigned ichan=0; ichan < skfb_nchan; ichan++)
    {
      V_p0 =  skfb[ichan*2];
      V_p1 =  skfb[ichan*2+1];

      if ( V_p0 > upper_thresh || V_p0 < lower_thresh || 
           V_p1 > upper_thresh || V_p1 < lower_thresh )
      {

        zap_chan_count++;

        float * zerop0 = output->get_datptr(ichan, 0) + (ddfb_start_idat*output_ndim);
        float * zerop1 = output->get_datptr(ichan, 1) +( ddfb_start_idat*output_ndim);

        for (unsigned j=0; j<ddfb_nsamples*output_ndim; j++)
        {
          zerop0[j] = 0;
          zerop1[j] = 0;
        }
      }
      // if any channel goes over the mega threshold, zap them all
      if ( V_p0 > mega_upper_thresh || V_p0 < mega_lower_thresh ||
           V_p1 > mega_upper_thresh || V_p1 < mega_lower_thresh )
      {
        zap_all_chan ++;
      }
    }

    // if enough data in this timeslice are corrupt, zap the lot
    if (zap_all_chan > 30)
    {

      if (verbose || debugd < 1)
        cerr << "dsp::ZapWeight::transformation MEGA threshold triggered affected "
             << "chans=" << zap_all_chan << " idat=" << idat << " total=" 
             << total_idats << endl;

      for (unsigned ichan=0; ichan<skfb_nchan; ichan++)
      {

        float * zerop0 = output->get_datptr(ichan, 0) + (ddfb_start_idat*output_ndim);
        float * zerop1 = output->get_datptr(ichan, 1) + (ddfb_start_idat*output_ndim);

        for (unsigned j=0; j<ddfb_nsamples*output_ndim; j++)
        { 
          zerop0[j] = 0; 
          zerop1[j] = 0; 
        }

      }
    }

    if (zap_chan_count)
    {
      zap_count += zap_chan_count;
      total_zaps += zap_chan_count;
    }

    if (verbose)
      cerr << "dsp::ZapWeight::transformation idat=" << idat << ", total_idats=" 
           << total_idats << ", zap_chan_count=" << zap_chan_count 
           << ", zap_all_chan=" << zap_all_chan << endl;

    skfb += skfb_nchan * skfb_npol;

    total_idats++;
  }

  if (debugd < 1)
    cerr << "dsp::ZapWeight::transformation idats now processed=" 
         << total_idats << ", zapped " << zap_count << " of " 
         << (skfb_ndat * skfb_nchan) << " dat pts" << ", total_zaps="
         << total_zaps << endl;

  // adjust next start sample for the SKFB (using input buffering)
  if (has_buffering_policy())
  {
    uint64_t skfb_next_sample = (ddfb_offset_samples + ddfb_ndat) / M;
    get_buffering_policy()->set_next_start (skfb_next_sample);
    if (debugd < 1)
      cerr << "dsp::ZapWeight::transformation next start sample=" << skfb_next_sample << endl;
  }

  if (debugd < 1)
    debugd++;
}

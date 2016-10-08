/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SKMasker.h"

#include "dsp/TimeSeries.h"
#include "dsp/BitSeries.h"
#include "dsp/IOManager.h"
#include "dsp/Unpacker.h"
#include "dsp/Input.h"

#include <assert.h>

using namespace std;

dsp::SKMasker::SKMasker () 
  : Transformation<TimeSeries,TimeSeries>("SKMasker",inplace)
{
  M = 0;
  ddfb_rate = 0;
  mask_rate = 0;
  total_idats = 0;
  debugd = 1;
}

dsp::SKMasker::~SKMasker ()
{
  if (verbose)
    cerr << "dsp::SKMasker::~SKMasker()" << endl;
}

void dsp::SKMasker::set_engine (Engine * _engine)
{
  engine = _engine;
}

void dsp::SKMasker::set_mask_input (BitSeries * _mask_input)
{
  mask_input = _mask_input;
}

void dsp::SKMasker::set_M (unsigned _M)
{
  M = _M;
}

//! Perform the transformation on the input time series
void dsp::SKMasker::transformation ()
{

  if (verbose || debugd < 1)
    cerr << "dsp::SKMasker::transformation input ndat="
         << input->get_ndat() << " ndim=" << input->get_ndim()
         << " nchan=" << input->get_nchan() << endl;

  ddfb_rate = input->get_rate();
  mask_rate = mask_input->get_rate();

  const unsigned ddfb_nchan = input->get_nchan();
  const uint64_t ddfb_ndat  = input->get_ndat();
  const uint64_t ddfb_npol  = input->get_npol();
  const unsigned ddfb_ndim  = input->get_ndim();

  const unsigned mask_nchan = mask_input->get_nchan();
  const unsigned mask_npol  = mask_input->get_npol();
  int64_t mask_ndat  = mask_input->get_ndat();

  const uint64_t ddfb_input_sample = input->get_input_sample();
  const uint64_t mask_input_sample = mask_input->get_input_sample();

  if (mask_npol != 1)
    throw Error (InvalidParam, "dsp::SKMasker::transformation",
                 "mask_npol != 1");
    
  if (debugd < 1)
  {
    cerr << "dsp::SKMasker::transformation DDFB: nchan=" << ddfb_nchan
         << " npol=" << ddfb_npol << " ndat=" << ddfb_ndat << " samples[" 
         << ddfb_input_sample << " - " << (ddfb_input_sample + ddfb_ndat) 
         << "]" << endl;
    cerr << "dsp::SKMasker::transformation MASK: nchan=" << mask_nchan
         << " npol=" << mask_npol << " ndat=" << mask_ndat << " samples["
         << mask_input_sample << " - " << (mask_input_sample + mask_ndat) 
         << "]" << endl;
  }

  // indicate the output timeseries contains zeroed data
  output->set_zeroed_data (true);

  // resize the output to ensure the hits array is reallocated
  if (engine)
  {
    if (verbose)
      cerr << "dsp::SKMasker::transformation output->resize(" << output->get_ndat() << ")" << endl;
    output->resize (output->get_ndat());
  }

  // get base pointer to mask bitseries
  unsigned char * mask = mask_input->get_datptr ();

  // calculate the offset between the ddfb and mask in ddfb samples
  MJD offset_mjd = input->get_start_time() - mask_input->get_start_time();

  int64_t ddfb_offset_samples = (int64_t) (offset_mjd.in_seconds() * ddfb_rate + 0.5);

  if (ddfb_offset_samples < 0)
  {
    throw Error (InvalidParam, "dsp::SKMasker::transformation", 
                 "ddfb_offset_samples < 0");
    return;
  }

  int64_t ddfb_idat   = ddfb_offset_samples % M;
  int64_t ddfb_offset = ddfb_offset_samples / M;

  if (debugd < 1)
    cerr << "dsp::SKMasker::transformation DDFB"
         << " offset_samples= " << ddfb_offset_samples 
         << " ddfb_idat=" << ddfb_idat 
         << " ddfb_offset=" << ddfb_offset << endl;

  // if the DDFB is off by more that 1 SKFB mask idat, input sample 
  // is more than 1 MASK block, adjust MASK ptr and ndat
  if (ddfb_offset) 
  {
    if (verbose)
      cerr << "dsp::SKMasker::transformation ddfb_offset > 0, adjusting mask" << endl;
    mask += ddfb_offset * mask_nchan * mask_npol;
    mask_ndat -= ddfb_offset;
  }

  // determine the number of mask blocks required to process the ddfb_ndat
  uint64_t mask_ndat_needed = (ddfb_ndat) / M;
  if ((ddfb_ndat) % M)
  {
    mask_ndat_needed++;
  }

  // truncate ddfb if there 
  if (mask_ndat_needed > mask_ndat)
  {
    cerr << "dsp::SKMasker::transformation more mask needed=" << mask_ndat_needed
         << " have=" << mask_ndat << endl;
    output->set_ndat (mask_ndat * M);
    cerr << "dsp::SKMasker::transformation truncating DDFB ndat to " << (mask_ndat * M) << endl;
  }

  if (debugd < 1)
    cerr << "dsp::SKMasker::transformation ddfb_ndat+ddfb+idat/M=" << ((ddfb_ndat + ddfb_idat) / M) 
         << " ddfb_ndat+ddfb_idat%M=" << ((ddfb_ndat + ddfb_idat) % M) << endl;

  if (mask_ndat_needed < mask_ndat)
  {
    if (verbose)
      cerr << "dsp::SKMasker::transformation limiting mask_ndat to " << mask_ndat_needed << ", was " << mask_ndat << endl;
    mask_ndat = mask_ndat_needed;
  }

  // the number of samples ddfb samples are offset from ddfb->get_datptr()
  uint64_t ddfb_start_idat;
  uint64_t ddfb_nsamples;
  uint64_t ddfb_end_idat;

  if (engine)
  {
    if (verbose)
      cerr << "dsp::SKMasker::transformation engine->setup()" << endl;
    engine->setup ();
  }

  for (uint64_t idat=0; idat < mask_ndat; idat++)
  {
    ddfb_start_idat = idat * M;
    ddfb_nsamples   = M;
    ddfb_end_idat   = ddfb_start_idat + ddfb_nsamples;

    if (verbose)
      cerr << "["<<idat<<" / " << mask_ndat << "] PRE  " << ddfb_start_idat << " -> " << ddfb_end_idat << "[" << ddfb_nsamples << "]" << endl;

    // if we are on the first idat, 
    if (ddfb_idat > ddfb_start_idat)
    {
      ddfb_nsamples -= ddfb_idat;
      ddfb_end_idat -= ddfb_idat;
    }
    else
    {
      ddfb_start_idat -= ddfb_idat;
      ddfb_end_idat -= ddfb_idat;
    }

    // special case for final block
    if (ddfb_end_idat > ddfb_ndat || (idat == mask_ndat -1))
    {
      ddfb_nsamples = ddfb_ndat - ddfb_start_idat;
      ddfb_end_idat = ddfb_ndat;
    }

    // the DDFB does not span this MASK idat
    if (ddfb_start_idat >= ddfb_ndat)
    {
      cerr << "dsp::SKMasker::transformation BREAK!!! idat=" << idat 
           << " ddfb_start_idat[" << ddfb_start_idat << "] > ddfb_ndat [" 
           << ddfb_ndat << "]" << endl;
      break;
    }

    if (verbose)
      cerr << "["<<idat<<" / " << mask_ndat << "] POST " << ddfb_start_idat << " -> " 
           << ddfb_end_idat << "[" << ddfb_nsamples << "]" << endl;

    if (engine) 
    {
      unsigned mask_offset = mask_nchan * mask_npol * idat;
      unsigned offset      = ddfb_start_idat*ddfb_ndim;
      unsigned end         = ddfb_nsamples*ddfb_ndim;

      //engine->perform (mask_input, mask_offset, output, offset, end);
    }
    else
    {
      // for each channel/pol in the SK FB
      for (unsigned ichan=0; ichan < mask_nchan; ichan++)
      {
        if (mask[ichan])
        {
          for (unsigned ipol=0; ipol < ddfb_npol; ipol++)
          {
            float * zero = output->get_datptr(ichan, ipol) + (ddfb_start_idat*ddfb_ndim);
            for (unsigned j=0; j<ddfb_nsamples*ddfb_ndim; j++)
              zero[j] = 0;
          }
        }
      }
      mask += mask_nchan * mask_npol;
    }

    total_idats++;
  }

  if (debugd < 1)
    debugd++;
}

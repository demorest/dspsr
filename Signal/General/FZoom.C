/***************************************************************************
 *
 *   Copyright (C) 2015 by Matthew Kerr
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/FZoom.h"
#include "Error.h"
#include <assert.h>

using namespace std;

dsp::FZoom::FZoom () 
  : Transformation <TimeSeries, TimeSeries> ("FZoom", outofplace)
  , centre_frequency(0)
  , bandwidth(0)
{
}

void dsp::FZoom::set_centre_frequency( double freq )
{
  centre_frequency = freq;
}

void dsp::FZoom::set_bandwidth( double bw )
{
  // Set bandwidth
  bandwidth = bw;

}

double dsp::FZoom::get_centre_frequency() const
{
  return centre_frequency;
}

double dsp::FZoom::get_bandwidth() const
{
  return bandwidth;
}

void dsp::FZoom::set_bounds()
{
  // Determine which channels lie within  bounds and set internal members
  double bw = input->get_bandwidth();
  if (bw < 0) { bw = -bw; }
  double in_freq = input->get_centre_frequency();
  if ( (centre_frequency > (in_freq+0.5*bw)) || 
       (centre_frequency < (in_freq-0.5*bw)) ) {
    throw Error (InvalidRange,"dsp::FZoom::set_bounds",
	    "requested centre frequency %.4f lies outside of data bounds", centre_frequency);
  }

  chan_lo=input->get_nchan();
  chan_hi=0;
  double freq_lo = centre_frequency - 0.5*bandwidth;
  double freq_hi = centre_frequency + 0.5*bandwidth;
  double half_chan_bw = 0.5*bw/input->get_nchan();
  for (unsigned ichan=0; ichan < input->get_nchan(); ++ichan) {
    double chan_freq = input->get_centre_frequency(ichan);
    double chan_freq_lo = chan_freq - half_chan_bw;
    double chan_freq_hi = chan_freq + half_chan_bw;
    if (chan_freq_hi < freq_lo) {
      continue;
    }
    if (chan_freq_lo > freq_hi) {
      continue;
    }
    if ( (chan_freq_lo > freq_lo) || (chan_freq_hi < freq_hi) ) {
      chan_lo = min(ichan,chan_lo);
      chan_hi = max(ichan,chan_hi);
    }
  }
  if (verbose) {
    cerr<<"dsp::Fzoom::set_bounds selected channels / frequencies: "<<endl<<
    "lo: " << chan_lo<< " / "<< input->get_centre_frequency(chan_lo)<<endl<<
    "hi: " << chan_hi<< " / "<< input->get_centre_frequency(chan_hi)<<endl;
  }
}

void dsp::FZoom::prepare ()
{
  // Set up output ahead of time -- necessary?
  get_output()->copy_configuration (get_input()) ;
}

void dsp::FZoom::transformation ()
{
  get_output()->copy_configuration (get_input()) ;
  set_bounds();
  get_output()->set_nchan(chan_hi-chan_lo+1);
  get_output()->set_centre_frequency(
      0.5*(input->get_centre_frequency(chan_lo) + 
           input->get_centre_frequency(chan_hi)));
  get_output()->set_bandwidth(get_input()->get_bandwidth()/get_input()->get_nchan()*get_output()->get_nchan());
  get_output()->resize(input->get_ndat());

  switch (input->get_order())
  {
    case TimeSeries::OrderFPT:
      fpt_copy ();
      break;

    case TimeSeries::OrderTFP:
      tfp_copy ();
      break;
  }

  get_output()->set_input_sample(get_input()->get_input_sample());

}

void dsp::FZoom::fpt_copy ()
{
  unsigned npol = input->get_npol();
  unsigned nchan = chan_hi - chan_lo + 1;
  uint64_t nfloat = input->get_ndat()*input->get_ndim();

  if (verbose)
    cerr << "dsp::FZoom::fpt_copy"
         << " input channel lo=" << chan_lo
         << " input channel hi=" << chan_hi
         << " nfloat=" << nfloat << endl;

  for (unsigned ichan=0; ichan < nchan; ++ichan) {
    for (unsigned ipol=0; ipol < npol; ++ipol) {
      const float* in = input->get_datptr(ichan+chan_lo,ipol);
      float* out = output->get_datptr(ichan,ipol);
      for (unsigned idat=0; idat < nfloat; ++idat) {
        out[idat] = in[idat];
      }
    }
  }
}

void dsp::FZoom::tfp_copy ()
{
  const unsigned npol = input->get_npol();
  const unsigned ndim = input->get_ndim();
  const unsigned nchan_in = input->get_nchan();
  const unsigned nchan_out = output->get_nchan();
  const uint64_t ndat = input->get_ndat();

  // floats per time sample
  unsigned nfloat_in = npol * ndim * nchan_in;
  unsigned nfloat_out = npol * ndim * nchan_out;

  if (verbose)
    cerr << "dsp::FZoom::tfp_copy"
         << " input channel lo=" << chan_lo
         << " input channel hi=" << chan_hi
         << " nfloat_in=" << nfloat_in
         << " nfloat_out=" << nfloat_out << endl;

  // input offset to correct start channel
  unsigned offset = npol * ndim * chan_lo;

  // block copy channels, pols & dim for each time sample
  for (unsigned idat=0; idat < ndat; ++idat) {

    const float* in = input->get_dattfp() + idat*nfloat_in + offset;
    float* out = output->get_dattfp() + idat*nfloat_out;

    for (unsigned ifloat=0; ifloat < nfloat_out; ++ifloat) {
      out[ifloat] = in[ifloat];
    }
  }

}

#ifdef D0
dsp::PhFZoom::PhFZoom () 
  : Transformation <PhaseSeries, PhaseSeries> ("PhFZoom", outofplace)
{
}

// perform TimeSeries transformation, then take care of hits
void dsp::PhFZoom::transformation () {
  dsp::FZoom::transformation ();
  // at this point, the _data_ have been zoomed, need only to update hits
  // NB hits array is [nchan , nbin]
  PhaseSeries* input = input;
  PhaseSeries* output = output;
  unsigned nbin = output->get_nbin();
  unsigned nchan = output->get_nchan();
  output->resize_hits(nbin);
  dsp::FZoom::cerr << "nchan= " << nchan << " chanlo=" << chan_lo << " chanhi=" << chan_hi << endl;
  unsigned* in = input->get_hits(chan_lo);
  unsigned* out = output->get_hits(0);
  for (unsigned i=0; i< nbin*nchan; ++i) {
    *out++ = *in++;
  }
}
#endif


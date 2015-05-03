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
  : Transformation <TimeSeries, TimeSeries> ("FZoom", anyplace)
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

void dsp::FZoom::set_engine( Engine* _engine )
{
  engine = _engine;
}

void dsp::FZoom::set_bounds()
{
  set_channel_bounds (
      input.get(),centre_frequency,bandwidth,&chan_lo,&chan_hi);
  if (verbose) {
    cerr<<"dsp::Fzoom::set_bounds selected channels / frequencies: "<<endl<<
    "lo: " << chan_lo<< " / "<< input->get_centre_frequency(chan_lo)<<endl<<
    "hi: " << chan_hi<< " / "<< input->get_centre_frequency(chan_hi)<<endl;
  }
}

void dsp::FZoom::prepare ()
{
  // Set up output ahead of time -- necessary?
  if ( get_input() != get_output() )
    get_output()->copy_configuration (get_input()) ;
}

void dsp::FZoom::transformation ()
{
  if (verbose)
    cerr << "dsp::FZoom:transformation input_sample= " << 
      input->get_input_sample () << " ndat=" << input->get_ndat() << endl;

  // TODO -- support out-of-order Observations, but not yet
  if (get_input()->get_nsub_swap())
    throw Error (InvalidState,"dsp::FZoom::transformation",
        "does not support sub-band swapped data (yet)");

  bool inplace = get_output() == get_input();

  if (inplace && engine)
    throw Error (InvalidState,"dsp::FZoom::transformation",
        "does not support in-place transformations on GPU");

  // set chan_lo and chan_hi
  set_bounds ();

  // if in place, do really inefficiently :(
  TimeSeries* dest = inplace? get_input()->clone() : get_output();

  // adjust centre frequency for output 
  dest->copy_configuration (get_input()) ;
  assert(dest->get_order () == input->get_order () );
  unsigned input_nchan = input->get_nchan ();
  double input_chanbw = input->get_bandwidth() / input_nchan;
  double df = 0.5*input_chanbw*(int(chan_lo)-int(input_nchan-chan_hi-1));
  dest->set_nchan (chan_hi-chan_lo+1);
  dest->set_centre_frequency (input->get_centre_frequency() + df);
  dest->set_bandwidth( input_chanbw * dest->get_nchan() );
  dest->resize (input->get_ndat());

  assert (input->get_centre_frequency(chan_lo) == 
           dest->get_centre_frequency(0));

  assert (input->get_centre_frequency(chan_hi) == 
           dest->get_centre_frequency(dest->get_nchan()-1));

  switch (input->get_order ())
  {
    case TimeSeries::OrderFPT:
    {
      if (engine)
        engine->fpt_copy(get_input(),dest,chan_lo,chan_hi);
      else
        fpt_copy (dest);
      break;
    }

    case TimeSeries::OrderTFP:
    {
      if (engine)
        throw Error (InvalidState,"dsp::FZoom::transformation",
            "does not support in-place transformations on GPU");
      else
        tfp_copy (dest);
      break;
    }

    default:
      throw Error (InvalidState, "dsp::FZoom::transformation",
          "unsupported data order");
  }

  if (inplace)
    get_output()->operator= (*dest);

  get_output()->set_input_sample (get_input()->get_input_sample());

}

void dsp::FZoom::fpt_copy (TimeSeries* dest)
{
  unsigned npol = input->get_npol();
  unsigned nchan = chan_hi - chan_lo + 1;
  uint64_t nfloat = input->get_ndat()*input->get_ndim(); // floats per chan

  if (verbose)
    cerr << "dsp::FZoom::fpt_copy"
         << " input channel lo=" << chan_lo
         << " input channel hi=" << chan_hi
         << " nfloat=" << nfloat << endl;

  for (unsigned ichan=0; ichan < nchan; ++ichan) {
    for (unsigned ipol=0; ipol < npol; ++ipol) {
      const float* in = input->get_datptr (ichan+chan_lo,ipol);
      float* out = dest->get_datptr (ichan,ipol);
      for (unsigned idat=0; idat < nfloat; ++idat) {
        out[idat] = in[idat];
      }
    }
  }
}

void dsp::FZoom::tfp_copy (TimeSeries* dest)
{
  const unsigned npol = input->get_npol();
  const unsigned ndim = input->get_ndim();
  const unsigned nchan_in = input->get_nchan();
  const unsigned nchan_out = dest->get_nchan();
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
    float* out = dest->get_dattfp() + idat*nfloat_out;

    for (unsigned ifloat=0; ifloat < nfloat_out; ++ifloat) {
      out[ifloat] = in[ifloat];
    }
  }

}

void dsp::FZoom::set_channel_bounds(const Observation* input,
    double centre_frequency, double bandwidth,
    unsigned* chan_lo, unsigned* chan_hi)
{
  // Determine which channels lie within  bounds and set internal members
  double bw = input->get_bandwidth();
  if (bw < 0) { bw = -bw; }
  double in_freq = input->get_centre_frequency();
  if ( ( (centre_frequency + 0.5*bandwidth) > (in_freq+0.5*bw) ) || 
       ( (centre_frequency - 0.5*bandwidth) < (in_freq-0.5*bw) ) ) {
    throw Error (InvalidRange,"dsp::FZoom::set_bounds",
	    "requested zoom band %.4f not a subset of data band", 
      centre_frequency);
  }

  *chan_lo=input->get_nchan();
  *chan_hi=0;
  double freq_lo = centre_frequency - 0.5*bandwidth,
         freq_hi = centre_frequency + 0.5*bandwidth,
         half_chan_bw = 0.5*bw/input->get_nchan();
  for (unsigned ichan=0; ichan < input->get_nchan(); ++ichan) {
    double chan_freq = input->get_centre_frequency(ichan),
           chan_freq_lo = chan_freq - half_chan_bw,
           chan_freq_hi = chan_freq + half_chan_bw;
    if ( (chan_freq_hi < freq_lo) || (chan_freq_lo > freq_hi) )
      continue;
    if ( (chan_freq_lo > freq_lo) || (chan_freq_hi < freq_hi) ) {
      *chan_lo = min(ichan,*chan_lo);
      *chan_hi = max(ichan,*chan_hi);
    }
  }
}

void dsp::FZoom::Engine::set_direction(
    dsp::FZoom::Engine::Direction _direction)
{
  direction = _direction;
}


/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/OptimalFFT.h"
#include "Pulsar/Config.h"

#include <math.h>

using namespace std;

bool dsp::OptimalFFT::verbose = true;

dsp::OptimalFFT::OptimalFFT ()
{
  nchan = 1;
  simultaneous = false;
}

void dsp::OptimalFFT::set_nchan (unsigned n)
{
  nchan = n;
}

//! Set true when convolution is performed during filterbank synthesis
void dsp::OptimalFFT::set_simultaneous (bool flag)
{
  simultaneous = flag;
}

/*! This method computes the optimal FFT length to use when performing
    multiple FFTs on a long time series.  If FFT cost measurements
    are not available, this function assumes that the FFT is an
    O(NlogN) operation.

   GIVEN:
   nchan      - number of channels into which data will be divided
   nfilt      - filter length (points discarded from each FFT in each channel)
   
   RETURNS:
   The return value is the optimal FFT length, nfft

   -----------------------------------------------------------------------

   Where ngood = nfft - nfilt, the following ideas are held:
   
   1) performance is better if ngood is a large fraction of nfft,
   2) FFT performance is much better with smaller nfft.

   The timescale for one FFT is proportional to NlogN, or:

                  t = nfft * log(nfft)

   The number of FFTs performed on a segment of M time samples is:

                 Nf = M / (nfft - nfilt)

   The total time spent on FFTs is then:

                  T = t * Nf = nfft * log(nfft) * M/(nfft-nfilt)

   Where M may be considered constant, and nfilt is given,
   this function aims to minimize:

            T(nfft) = nfft * log(nfft) / (nfft-nfilt)

   *********************************************************************** */

unsigned dsp::OptimalFFT::get_nfft (unsigned nfilt) const
{
  if (!nfilt)
    throw Error (InvalidParam, "dsp::OptimalFFT::get_nfft", "nfilt == 0");

  // the smallest FFT possible, given nfilt, is the next power of two higher
  unsigned nfft_min = (unsigned) pow (2.0, ceil(log2(nfilt))); 

  unsigned best_nfft = nfft_min;
  double best_cost = 0;

  unsigned theory_nfft = nfft_min;
  double theory_cost = 0;

  if (!bench)
  {
    bench = new FTransform::Bench;
    bench->set_path( Pulsar::Config::get_runtime() );
  }

  unsigned nfft_max = bench->get_max_nfft ();

  for (unsigned nfft = nfft_min; nfft <= nfft_max; nfft *= 2)
  {
    double cost = compute_cost (nfft, nfilt);
    double theory = nfft * log2(nfft) / (nfft-nfilt);

    if (verbose)
      cerr << "NFFT " << nfft << "  %kept=" << double(nfft-nfilt)/nfft * 100.0 
	   << " theory=" << theory << " bench=" << cost << endl;

    if (cost < best_cost || best_cost == 0)
    {
      best_cost = cost;
      best_nfft = nfft;
    }

    if (theory < theory_cost || theory_cost == 0)
    {
      theory_cost = theory;
      theory_nfft = nfft;
    }
  }

  cerr << "best=" << best_nfft << endl;
  cerr << "theory=" << theory_nfft << endl;

  return best_nfft;
}

double dsp::OptimalFFT::compute_cost (unsigned nfft, unsigned nfilt) const
{
  FTransform::Bench::verbose = verbose;

  if (nchan == 1)
    return bench->get_best( nfft ).cost / (nfft - nfilt);

  double total = 0.0;

  if (simultaneous)
  {
    double fwd = bench->get_best( nfft*nchan ).cost;
    double bwd = nchan * bench->get_best( nfft ).cost;
    total = fwd + bwd;
  }
  else
  {
    double fbank = bench->get_best( nchan ).cost;
    double conv = 2 * nchan * bench->get_best( nfft ).cost;
    total = fbank + conv;
  }

  return total / ((nfft - nfilt) * nchan);
}

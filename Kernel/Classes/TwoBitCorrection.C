#include <iostream>

#include <assert.h>
#include <math.h>

#include "TwoBitCorrection.h"
#include "genutil.h"

/*! From Jenet&Anderson "The Effects of Digitization on Nonstationary
  Stochastic Signals with Applications to Pulsar Signal Baseband
  Recording", Table 1 */
const double dsp::TwoBitCorrection::optimal_2bit_threshold = 0.9674;

//! Null constructor
dsp::TwoBitCorrection::TwoBitCorrection (const char* _name, Behaviour _type)
  : Operation (_name, _type)
{
  nchannel  = -1;
  maxstates = -1;

  histograms = NULL;
  dls_lookup = NULL;

  nsample = 0;
  cutoff_sigma = 0.0;
  n_min = 0;
  n_max = 0;
}

void dsp::TwoBitCorrection::destroy ()
{
  if (dls_lookup != NULL) delete [] dls_lookup; dls_lookup = NULL;
  if (histograms != NULL) delete [] histograms; histograms = NULL;
}

void dsp::TwoBitCorrection::set_nchannel (int chan)
{
  if (chan != nchannel)
    destroy();
  nchannel = chan;
}

void dsp::TwoBitCorrection::set_nsample (int ppwt)
{
  if (ppwt != nsample)
    destroy();
  nsample = ppwt;
}

void dsp::TwoBitCorrection::set_cutoff_sigma (float cosig)
{
  if (cutoff_sigma != cosig)
    destroy();
  cutoff_sigma = cosig;
}

//
// this function allocates arrays of the appropriate size
// the following data members must be initialized before calling this
// function:
// nsample, n_min, n_max, maxstates, nchannel
//
void dsp::TwoBitCorrection::allocate ()
{
  if (verbose) cerr << "TwoBitCorrection::allocate enter\n";

  int n_range = n_max - n_min + 1;

#ifdef _DEBUG
  cerr << "TwoBitCorrection::allocate"
       << "\n range     = " << n_range
       << "\n maxstates = " << maxstates
       << "\n nsample  = " << nsample
       << "\n nchannel  = " << nchannel
       << endl;
#endif

  if (n_range < 2)
    throw_str ("TwoBitCorrection::allocate invalid cutoff_sigma=%f"
		" nbin=%d nmax=%d", cutoff_sigma, n_min, n_max); 

  destroy ();

  if (verbose) cerr << "TwoBitCorrection::allocate allocate buffers\n";
  dls_lookup = new float [n_range * maxstates];
  assert (dls_lookup != 0);

  histograms = new unsigned long [nsample * nchannel];
  assert (histograms != 0);

  zero_histogram ();

  if (verbose) cerr << "TwoBitCorrection::allocate exits\n";
}

void dsp::TwoBitCorrection::set_twobit_limits (int ppwt, float co_sigma)
{
  nsample = ppwt;
  cutoff_sigma = co_sigma;

  set_twobit_limits ();
}

void dsp::TwoBitCorrection::set_twobit_limits ()
{
  /* average number of samples that lay within the thresholds, x2 and x4,
     where -x2 = x4 = 0.9674 of the noise power, as in Table 1 of 
     Jenet&Anderson.  Apply t=0.9674sigma to Equation 45, 
     (or -xl=xh=0.9674 to Equation A2) to get: */
  float fraction_ones = erf(optimal_2bit_threshold/sqrt(2.0));

  float n_ave = float(nsample) * fraction_ones;
  // the root mean square deviation
  float n_sigma = sqrt(nsample);

  n_min = int (n_ave - (cutoff_sigma * n_sigma));
  n_max = int (n_ave + (cutoff_sigma * n_sigma));

  //fprintf (stderr, "TwoBitCorrection::set_twobit_limits n_min:%d n_max:%d\n",
  //n_min, n_max);

  if (n_max >= nsample) {
    if (verbose)
      cerr << "TwoBitCorrection::set_twobit_limits resetting nmax:"
	   << n_max << " to nsample-2:" << nsample-1 << endl;
    n_max = nsample-1;
  }

  if (n_min < 1) {
    if (verbose)
      cerr << "TwoBitCorrection::set_twobit_limits resetting nmin:"
	   << n_min << " to one:1" << endl;
    n_min = 1;
  }

  if (verbose) cerr << "TwoBitCorrection::set_twobit_limits nmin:"
		    << n_min << " and nmax:" << n_max << endl;

}

void dsp::TwoBitCorrection::zero_histogram ()
{
  if (!histograms)
    return;

  int nbins = nsample * nchannel;
  for (int ibin=0; ibin<nbins; ibin++)
    histograms [ibin] = 0;
}

double dsp::TwoBitCorrection::get_histogram_mean (int channel)
{
  if (channel < 0 || channel >= nchannel)
    throw_str ("TwoBitCorrection::get_histogram_mean"
		" invalid channel=%d", channel);

  double ones = 0.0;
  double pts  = 0.0;

  unsigned long* hist = histograms + channel*nsample;
  for (int ival=0; ival<nsample; ival++) {
    double samples = double (hist[ival]);
    ones += samples * double (ival);
    pts  += samples * double (nsample);
  }
  return ones/pts;
}

//! Calculate the mean voltage and power from Bit_Stream data
int64 dsp::TwoBitCorrection::stats (vector<double>& mean,
				    vector<double>& power)
{
  cerr << "TwoBitCorrection::stats not implemented" << endl;
  return -1;
}

#include <iostream>
#include <assert.h>
#include <math.h>

#include "TwoBitCorrection.h"
#include "Timeseries.h"
#include "genutil.h"
#include "ierf.h"

/*! From JA98, Table 1 */
const double dsp::TwoBitCorrection::optimal_2bit_threshold = 0.9674;

//! Null constructor
dsp::TwoBitCorrection::TwoBitCorrection (const char* _name, Behaviour _type)
  : Operation (_name, _type)
{
  nchannel  = -1;

  histograms = NULL;
  dls_lookup = NULL;

  nsample = 0;
  cutoff_sigma = 0.0;
  n_min = 0;
  n_max = 0;
}

//! Initialize and resize the output before calling unpack
void dsp::TwoBitCorrection::operation ()
{
  if (input->get_nbit() != 2)
    throw_str ("TwoBitCorrection::operation input not 2-bit digitized");

  // set the Observation information
  output->Observation::operator=(*input);

  // output will contain floating point values
  output->set_nbit (8 * sizeof(float));

  // resize the output 
  output->resize (input->get_ndat());

  // call the virtual method defined by subclasses
  unpack ();
}

void dsp::TwoBitCorrection::destroy ()
{
  if (dls_lookup != NULL) delete [] dls_lookup; dls_lookup = NULL;
  if (histograms != NULL) delete [] histograms; histograms = NULL;
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
  dls_lookup = new float [n_range * 4];
  assert (dls_lookup != 0);

  histograms = new unsigned long [nsample * nchannel];
  assert (histograms != 0);

  zero_histogram ();

  if (verbose) cerr << "TwoBitCorrection::allocate exits\n";
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

/* *************************************************************************
   TwoBitCorrection::build

   Generates a lookup table of output levels: y1 -> y4, for the range of 
   sample-statistics within the specified cutoff_sigma.

   This table may then be used to employ the dynamic level setting technique
   described by Jenet&Anderson in "Effects of Digitization on Nonstationary
   Stochastic Signals" for data recorded with CBR or CPSR.

   Where possible, references are made to the equations given in this paper,
   which are mostly found in Section 6.
   ********************************************************************** */

void dsp::TwoBitCorrection::build (int nchan, int nsamp, float sigma)
{
  if (verbose) cerr << "TwoBitCorrection::build:: "
		    << " nsamp=" << nsamp
		    << " cutoff=" << sigma << "sigma\n";

  nchannel = nchan;
  nsample = nsamp;
  cutoff_sigma = sigma;

  set_twobit_limits ();
  
  float* dls_lut = dls_lookup;
  float lo=0, hi=0;

  for (int n_in=n_min; n_in <= n_max; n_in++)  {

    get_output_levels (n_in, lo, hi);

    for (unsigned char val=0; val<4; val++)  {
      if (get_hi(val))
	*dls_lut = get_sign(val) * hi;
      else
	*dls_lut = get_sign(val) * lo;
      dls_lut ++;
    }
  }
}

/*!  Given the number of digitized samples in the low voltage state,
  this method returns the optimal values for low and high output
  voltage states.

  \pre the nsample attribute must be set before calling this method
  \param nlo number of low voltage state samples (out of nsample)
  \retval lo the low voltage output state
  \retval hi the hi voltage output state
*/
  
void dsp::TwoBitCorrection::get_output_levels (int nlo, float& lo, float& hi) 
{
  static float root_pi = sqrt(M_PI);
  static float root2   = sqrt(2.0);

  if (nsample < 1)
    throw_str ("TwoBitCorrection::get_output_levels"
	       " invalid nsample=%d", nsample);

  if (nlo < 0 || nlo > nsample)
    throw_str ("TwoBitCorrection::get_output_levels"
	       " invalid nlo=%d", nlo);

  /* Refering to JA98, nlo is the number of samples between x2 and x4, 
     and p_in is the left-hand side of Eq.44 */
  float p_in = (float) nlo / (float) nsample;

  /* The inverse error function of p_in gives alpha, equal to the 
     "t/(root2 * sigma)" in brackets on the right-hand side of Eq.45 */
  float alpha = ierf (p_in);
  float expon = exp (-alpha*alpha);

  // Equation 41 (-y2, y3)
  lo = 2.0/(root2*alpha) * sqrt(1.0 - (2.0*alpha/root_pi)*(expon/p_in));

  // Equation 40 (-y1, y4)
  hi = 2.0/(root2*alpha) * sqrt(1.0 + (2.0*alpha/root_pi)*(expon/(1.0-p_in)));
}

bool dsp::TwoBitCorrection::get_hi (unsigned char val)
{
  bool hi [4] = { 1, 0, 0, 1 };
  return hi[val];
}

int dsp::TwoBitCorrection::get_sign (unsigned char val)
{
  float sign [4] = { -1.0, -1.0, 1.0, 1.0 };
  return sign[val];
}

//! Calculate the mean voltage and power from Bit_Stream data
int64 dsp::TwoBitCorrection::stats (vector<double>& mean,
				    vector<double>& power)
{
  cerr << "TwoBitCorrection::stats not implemented" << endl;
  return -1;
}

#include <iostream>
#include <assert.h>
#include <math.h>

#include "dsp/TwoBitCorrection.h"
#include "dsp/TwoBitTable.h"
#include "dsp/Timeseries.h"

#include "genutil.h"
#include "ierf.h"

/*! From JA98, Table 1 */
const double dsp::TwoBitCorrection::optimal_threshold = 0.9674;

bool dsp::TwoBitCorrection::keep_histogram = true;

//! Null constructor
dsp::TwoBitCorrection::TwoBitCorrection (const char* _name) : Unpacker (_name)
{
  // Sub-classes must define these three variables
  nchannel = -1;
  channels_per_byte = -1;

  // Sub-classes may re-define these
  nsample = 512;
  cutoff_sigma = 3.0;
  table = NULL;

  // These are set in set_limits()
  n_min = 0;
  n_max = 0;

  // This is set in build()
  built = false;
}

dsp::TwoBitCorrection::~TwoBitCorrection ()
{
}

//! Set the number of time samples used to estimate undigitized power
void dsp::TwoBitCorrection::set_nsample (int _nsample)
{
  if (nsample == _nsample)
    return;

  if (verbose)
    cerr << "TwoBitCorrection::set_nsample = " << _nsample << endl;

  nsample = _nsample;
  built = false;
}

//! Set the cut off power for impulsive interference excision
void dsp::TwoBitCorrection::set_cutoff_sigma (float _cutoff_sigma)
{
  if (cutoff_sigma == _cutoff_sigma)
    return;

  if (verbose)
    cerr << "TwoBitCorrection::set_cutoff_sigma = " << _cutoff_sigma << endl;

  cutoff_sigma = _cutoff_sigma;
  built = false;
}

void dsp::TwoBitCorrection::set_table (TwoBitTable* _table)
{
  if (table == _table)
    return;

  if (verbose)
    cerr << "TwoBitCorrection::set_table" << endl;

  table = _table;
  built = false;
}

//! Initialize and resize the output before calling unpack
void dsp::TwoBitCorrection::operation ()
{
  if (input->get_nbit() != 2)
    throw_str ("TwoBitCorrection::operation input not 2-bit digitized");

  if (verbose)
    cerr << "TwoBitCorrection::operation" << endl;;

  // build the two-bit lookup table
  if (!built)
    build ();

  Unpacker::operation ();
}


void dsp::TwoBitCorrection::set_limits ()
{
  if (verbose)
    cerr << "TwoBitCorrection::set_limits" << endl;;

  float fraction_ones = get_optimal_fraction_low();

  float n_ave = float(nsample) * fraction_ones;
  // the root mean square deviation
  float n_sigma = sqrt(nsample);

  n_min = int (n_ave - (cutoff_sigma * n_sigma));
  n_max = int (n_ave + (cutoff_sigma * n_sigma));

  if (n_max >= nsample) {
    if (verbose)
      cerr << "TwoBitCorrection::set_limits resetting nmax:"
	   << n_max << " to nsample-2:" << nsample-1 << endl;
    n_max = nsample-1;
  }

  if (n_min < 1) {
    if (verbose)
      cerr << "TwoBitCorrection::set_limits resetting nmin:"
	   << n_min << " to one:1" << endl;
    n_min = 1;
  }

  if (verbose) cerr << "TwoBitCorrection::set_limits nmin:"
		    << n_min << " and nmax:" << n_max << endl;
}

void dsp::TwoBitCorrection::zero_histogram ()
{
  if (verbose)
    cerr << "TwoBitCorrection::zero_histogram" << endl;;

  for (unsigned ichan=0; ichan < histograms.size(); ichan++)
    for (unsigned ibin=0; ibin<histograms[ichan].size(); ibin++)
      histograms[ichan][ibin] = 0;
}

double dsp::TwoBitCorrection::get_histogram_mean (int channel) const
{
  if (channel < 0 || channel >= nchannel)
    throw_str ("TwoBitCorrection::get_histogram_mean"
		" invalid channel=%d", channel);

  double ones = 0.0;
  double pts  = 0.0;

  for (int ival=0; ival<nsample; ival++) {
    double samples = double (histograms[channel][ival]);
    ones += samples * double (ival);
    pts  += samples * double (nsample);
  }
  return ones/pts;
}

unsigned long dsp::TwoBitCorrection::get_histogram_total (int channel) const
{
  unsigned long nweights = 0;

  for (int iwt=0; iwt<nsample; iwt++)
    nweights += histograms[channel][iwt];

  return nweights;
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

void dsp::TwoBitCorrection::build ()
{
  if (built)
    return;

  if (verbose) cerr << "TwoBitCorrection::build:: "
		    << " nsamp=" << nsample
		    << " cutoff=" << cutoff_sigma << "sigma\n";

  if (nchannel<1)
    throw_str ("TwoBitCorrection::build invalid nchannel=%d", nchannel);

  if (channels_per_byte<1)
    throw_str ("TwoBitCorrection::build invalid channels_per_byte=%d",
	       channels_per_byte);
 
  if (!table)
    throw_str ("TwoBitCorrection::build no TwoBitTable");

  set_limits ();

  int n_range = n_max - n_min + 1;

  bool huge = (channels_per_byte == 1);

  int size = TwoBitTable::vals_per_byte;
  if (huge)
    size *= TwoBitTable::unique_bytes;

  if (verbose) cerr << "TwoBitCorrection::build allocate buffers\n";
  dls_lookup.resize (n_range * size);

  generate (dls_lookup.begin(), 0, n_min, n_max, nsample, table, huge);

  histograms.resize (nchannel);
  for (int ichan=0; ichan < nchannel; ichan++)
    histograms[ichan].resize(nsample);

  zero_histogram ();

  nlo_lookup.resize (TwoBitTable::unique_bytes);

  // flatten the table again (precision errors cause mismatch of lo_valsq)
  table->set_lo_val (1.0);
  float lo_valsq = 1.0;

  for (unsigned byte = 0; byte < TwoBitTable::unique_bytes; byte++) {

    nlo_lookup[byte] = 0;
    const float* fourvals = table->get_four_vals (byte);

    for (unsigned ifv=0; ifv<TwoBitTable::vals_per_byte; ifv++)
      if (fourvals[ifv]*fourvals[ifv] == lo_valsq)
	nlo_lookup[byte] ++;

  }

  built = true;

  if (verbose) cerr << "TwoBitCorrection::build exits\n";

}

void dsp::TwoBitCorrection::generate (float* dls, float* spc, 
				      int n_min, int n_max, int n_tot,
				      TwoBitTable* table, bool huge)
{
  static float root_pi = sqrt(M_PI);
  static float root2   = sqrt(2.0);

  for (int nlo=n_min; nlo <= n_max; nlo++)  {
  
    /* Refering to JA98, nlo is the number of samples between x2 and x4, 
       and p_in is the left-hand side of Eq.44 */
    float p_in = (float) nlo / (float) n_tot;

    float lo, hi, A;
    output_levels (p_in, lo, hi, A);

    table->set_lo_val (lo);
    table->set_hi_val (hi);

    if (huge) {
      /* Generate the 256 sets of four output floating point values
	 corresponding to each byte */
      table->generate (dls);
      dls += TwoBitTable::unique_bytes * TwoBitTable::vals_per_byte;
    }
    else {
      // Generate the four output levels corresponding to each 2-bit number
      table->four_vals (dls);
      dls += TwoBitTable::vals_per_byte;
    }

    if (spc) {
      *spc = A;
      spc ++;
    }
  }
}

/*!  Given the fraction of digitized samples in the low voltage state,
  this method returns the optimal values for low and high output
  voltage states, as well as the fractional scattered power

  \param p_in fraction of low voltage state samples
  \retval lo the low voltage output state
  \retval hi the hi voltage output state
  \retval A the fractional scattered power
*/
void dsp::TwoBitCorrection::output_levels (float p_in, 
					   float& lo, float& hi, float& A) 
{
  static float root_pi = sqrt(M_PI);
  static float root2   = sqrt(2.0);

  /* Refering to JA98, p_in is the left-hand side of Eq.44, the
     fraction of samples between x2 and x4 */

  /* The inverse error function of p_in gives alpha, equal to the 
     "t/(root2 * sigma)" in brackets on the right-hand side of Eq.45 */
  float alpha = ierf (p_in);
  float expon = exp (-alpha*alpha);

  // Equation 41 (-y2, y3)
  lo = root2/alpha * sqrt(1.0 - (2.0*alpha/root_pi)*(expon/p_in));

  // Equation 40 (-y1, y4)
  hi = root2/alpha * sqrt(1.0 + (2.0*alpha/root_pi)*(expon/(1.0-p_in)));

  // Equation 43
  float halfrootnum = lo*(1-expon) + hi*expon;
  float num = 2.0 * halfrootnum * halfrootnum;
  A = num / ( M_PI * ((lo*lo-hi*hi)*p_in + hi*hi) );
}

/*! Return the average number of samples that lay within the
     thresholds, x2 and x4, where -x2 = x4 = 0.9674 of the noise
     power, as in Table 1 of JA98.  Apply t=0.9674sigma to
     Equation 45, (or -xl=xh=0.9674 to Equation A2) to get: */
float dsp::TwoBitCorrection::get_optimal_fraction_low () const
{
  return erf (optimal_threshold / sqrt(2.0));
}

void dsp::TwoBitCorrection::unpack ()
{
  if (input->get_state() != Signal::Nyquist)
    throw_str ("TwoBitCorrection::unpack input not real sampled");

  if (input->get_nbit() != 2)
    throw_str ("TwoBitCorrection::unpack input not 2-bit sampled");

  int64 ndat = input->get_ndat();

  if (ndat % TwoBitTable::vals_per_byte)
    throw_str ("TwoBitCorrection::unpack input ndat="I64" != 4n", ndat);
  
  if (ndat < nsample)
    throw_str ("TwoBitCorrection::unpack input ndat="I64" < nsample=%d",
	       ndat, nsample);

  const unsigned char* rawptr = input->get_rawptr();

  int npol = input->get_npol();

  for (int ipol=0; ipol<npol; ipol++) {

    const unsigned char* from = rawptr + ipol;

    float* into = output->get_datptr (0, ipol);

    unsigned long* hist = 0;

    if (keep_histogram)
      hist = histograms[ipol].begin();

    poln_unpack (into, from, ndat, hist, npol);
      
  }  // for each polarization

}

void dsp::TwoBitCorrection::poln_unpack (float* data,
					      const unsigned char* raw, 
					      uint64 ndat,
					      unsigned long* hist,
					      unsigned gap)
{
  // 4 floating-point samples per byte
  const unsigned samples_per_byte = TwoBitTable::vals_per_byte;

  // 4*256 floating-point samples for all unique bytes
  const unsigned lookup_block_size =
    samples_per_byte * TwoBitTable::unique_bytes;

  unsigned long n_weights = (unsigned long) ceil (float(ndat)/float(nsample));

  assert (n_weights*nsample >= ndat);

  unsigned long bytes_left = ndat / samples_per_byte;
  unsigned long bytes_per_weight = nsample / samples_per_byte;
  unsigned long bytes = bytes_per_weight;
  unsigned bt;
  unsigned pt;

  float* section = 0;
  float* fourval = 0;

  for (unsigned long wt=0; wt<n_weights; wt++) {

    const unsigned char* rawptr = raw;

    if (bytes > bytes_left) {
      rawptr -= (bytes_per_weight - bytes_left) * gap;
      bytes = bytes_left;
    }

    // calculate the weight based on the last nsample pts
    int n_lo = 0;
    for (bt=0; bt<bytes_per_weight; bt++) {
      n_lo += nlo_lookup [*rawptr];
      rawptr += gap;
    }

    if (hist)
      hist [n_lo] ++;

    if (n_lo<n_min || n_lo>n_max) {
      for (bt=0; bt<bytes*samples_per_byte; bt++) {
	*data = 0.0;
	data ++;
      }
    }

    else {

      rawptr = raw;
      section = dls_lookup.begin() + (n_lo-n_min) * lookup_block_size;

      for (bt=0; bt<bytes; bt++) {
	fourval = section + unsigned(*rawptr) * samples_per_byte;
	for (pt=0; pt<samples_per_byte; pt++) {
	  *data = fourval[pt];
	  data ++;
	}
	rawptr += gap;
      }

    }

    bytes_left -= bytes;
    raw += bytes * gap;
  }

}


int64 dsp::TwoBitCorrection::stats(vector<double>& m, vector<double>& p)
{
  static float* lu_sum = 0;
  static float* lu_sumsq = 0;
  static unsigned char* lu_nlo = 0;

  if (!input)
    throw_str ("TwoBitCorrection::stats no input");

  if (input->get_nbit() != 2)
    throw_str ("TwoBitCorrection::stats input nbit != 2");

  if (input->get_state() != Signal::Nyquist)
    throw_str ("TwoBitCorrection::stats input state != Nyquist");

  if (int(histograms.size()) != nchannel) {
    histograms.resize(nchannel);
    for (int ichan=0; ichan < nchannel; ichan++)
      histograms[ichan].resize (nsample);
  }

  if (lu_sum == NULL) {

    //if (verbose)
    cerr << "TwoBitCorrection::stats generate lookup table" <<endl;

    // calculate the sum and sum-squared of the four time samples in each byte
    lu_sum = new float [TwoBitTable::unique_bytes];
    lu_sumsq = new float [TwoBitTable::unique_bytes];
    lu_nlo = new unsigned char [TwoBitTable::unique_bytes];

    const float* fourvals = 0;
    float val = 0;
    float valsq = 0;
    float lo_valsq = table->get_lo_val() * table->get_lo_val();

    for (unsigned byte = 0; byte < TwoBitTable::unique_bytes; byte++) {
      
      lu_sum[byte] = 0;
      lu_sumsq[byte] = 0;
      lu_nlo[byte] = 0;

      fourvals = table->get_four_vals (byte);

      for (unsigned ifv=0; ifv<TwoBitTable::vals_per_byte; ifv++) {
	val = fourvals[ifv];
	valsq = val * val;

	lu_sum[byte] += val;
	lu_sumsq[byte] += valsq;

	if (valsq == lo_valsq)
	  lu_nlo[byte] ++;
      }
    }

  }

  // number of weight samples
  unsigned nweights = input->get_ndat() / nsample;

  unsigned npol = input->get_npol ();

  unsigned nbytes = input->nbytes() / npol;

  // number of bytes per weight sample
  unsigned nbytes_per_weight = nbytes / nweights;


  if (verbose)
    cerr << "TwoBitCorrection::stats npol=" << npol 
	 << " nweights=" << nweights << " bytespw=" << nbytes_per_weight
	 << endl;

  m.resize(npol);
  p.resize(npol);

  unsigned total_samples = 0;

  for (unsigned ipol=0; ipol < npol; ipol++) {

    unsigned long* hist = histograms[ipol].begin();

    double sum = 0;
    double sumsq = 0;
    total_samples = 0;

    const unsigned char* data = input->get_rawptr() + ipol;
    unsigned char datum;

    for (unsigned iwt=0; iwt<nweights; iwt++) {

      unsigned long nlo = 0;

      for (unsigned ipwt=0; ipwt < nbytes_per_weight; ipwt++) {

	datum = *data;
	data += npol;

	sum += lu_sum[datum];
	sumsq += lu_sumsq[datum];
	nlo += lu_nlo[datum];

	// every byte has four samples from one polarization in it
	total_samples += TwoBitTable::vals_per_byte;
      }

      if (hist)
	hist[nlo] ++;
    }
    //cerr << "sum:" << ipol << "=" << sum << endl;
    //cerr << "sumsq:" << ipol << "=" << sumsq << endl;
    m[ipol] = sum;
    p[ipol] = sumsq;
  }

  if (verbose)
    cerr << "TwoBitCorrection::stats return total samples " 
	 << total_samples <<  endl;

  // return the total number of timesamples measured
  return total_samples;
}


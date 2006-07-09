/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include <iostream>
#include <assert.h>
#include <math.h>

#include "dsp/TwoBitCorrection.h"
#include "dsp/TwoBitTable.h"
#include "dsp/WeightedTimeSeries.h"

#include "Error.h"
#include "ierf.h"

// #define _DEBUG 1

/*! From JA98, Table 1 */
const double dsp::TwoBitCorrection::optimal_threshold = 0.9674;

bool dsp::TwoBitCorrection::change_levels = true;

//! Null constructor
dsp::TwoBitCorrection::TwoBitCorrection (const char* _name) 
    : HistUnpacker (_name)
{
  // Sub-classes may re-define these
  set_nsample (512);
  set_ndig (2);

  if (psrdisp_compatible) {
    cerr << "dsp::TwoBitCorrection psrdisp compatibility\n"
      "   using cutoff sigma of 6.0 instead of 10.0" << endl;
    cutoff_sigma = 6.0;
  }
  else
    cutoff_sigma = 10.0;

  threshold = optimal_threshold;

  // Sub-classes must define this or set_table must be called
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

/*! By default, one polarization is output in one byte */
unsigned dsp::TwoBitCorrection::get_ndig_per_byte () const
{ 
  return 1;
}

/*! By default, the data from each polarization is interleaved byte by byte */
unsigned dsp::TwoBitCorrection::get_input_offset (unsigned idig) const
{
  return idig;
}

/*! By default, the data from each polarization is interleaved byte by byte */
unsigned dsp::TwoBitCorrection::get_input_incr () const 
{
  return 2;
}

/*! By default, the output from each digitizer is contiguous */
unsigned dsp::TwoBitCorrection::get_output_incr () const
{
  return 1;
}

//! Set the number of time samples used to estimate undigitized power
void dsp::TwoBitCorrection::set_nsample (unsigned _nsample)
{
  if (get_nsample() != _nsample)
    built = false;

  HistUnpacker::set_nsample (_nsample);
}

//! Set the cut off power for impulsive interference excision
void dsp::TwoBitCorrection::set_threshold (float _threshold)
{
  if (threshold == _threshold)
    return;

  if (verbose)
    cerr << "dsp::TwoBitCorrection::set_threshold = "<<_threshold<<endl;

  threshold = _threshold;
  built = false;
}

//! Set the cut off power for impulsive interference excision
void dsp::TwoBitCorrection::set_cutoff_sigma (float _cutoff_sigma)
{
  if (cutoff_sigma == _cutoff_sigma)
    return;

  if (verbose)
    cerr << "dsp::TwoBitCorrection::set_cutoff_sigma = "<<_cutoff_sigma<<endl;

  cutoff_sigma = _cutoff_sigma;
  built = false;
}

void dsp::TwoBitCorrection::set_table (TwoBitTable* _table)
{
  if (table.get() == _table)
    return;

  if (verbose)
    cerr << "dsp::TwoBitCorrection::set_table" << endl;

  table = _table;
  built = false;
}

//! Get the digitization convention
const dsp::TwoBitTable* dsp::TwoBitCorrection::get_table () const
{ 
  return table;
}

void dsp::TwoBitCorrection::set_output (TimeSeries* _output)
{
  if (verbose)
    cerr << "dsp::TwoBitCorrection::set_output (TimeSeries*)" << endl;

  Unpacker::set_output (_output);
  weighted_output = dynamic_cast<WeightedTimeSeries*> (_output);
}

//! Initialize and resize the output before calling unpack
void dsp::TwoBitCorrection::transformation ()
{
  if (verbose)
    cerr << "dsp::TwoBitCorrection::transformation" << endl;;

  if (input->get_nbit() != 2)
    throw Error (InvalidParam, "dsp::TwoBitCorrection::transformation",
		 "input not 2-bit digitized");

  // build the two-bit lookup table
  if (!built){
    if (verbose)
      cerr << "dsp::TwoBitCorrection::transformation calling build" << endl;
    build ();
  }

  // set the Observation information
  output->Observation::operator=(*input);

  if (weighted_output) {
    weighted_output -> set_ndat_per_weight (get_nsample());
    weighted_output -> set_nchan_weight (1);
    weighted_output -> set_npol_weight (input->get_npol());
  }

  // resize the output 
  output->resize (input->get_ndat());

  if (weighted_output)
    weighted_output -> neutral_weights ();

  if (verbose)
    cerr << "dsp::TwoBitCorrection::transformation calling unpack" << endl;

  // unpack the data
  unpack ();

  if (weighted_output) {

    weighted_output -> mask_weights ();
    uint64 nbad = weighted_output -> get_nzero ();
    discarded_weights += nbad;

    if (nbad && verbose)
      cerr << "dsp::TwoBitCorrection::transformation " << nbad 
           << "/" << weighted_output -> get_nweights()
           << " total bad weights" << endl;

  }

  if (verbose)
    cerr << "dsp::TwoBitCorrection::transformation exit" << endl;
}



void dsp::TwoBitCorrection::set_limits ()
{
  if (verbose)
    cerr << "dsp::TwoBitCorrection::set_limits" << endl;;

  float fraction_ones = get_fraction_low();

  // expectation value of the binomial distribution, JA98, Eqn. A6
  // cf. http://mathworld.wolfram.com/BinomialDistribution.html, Eqn. 9
  float nlo_mean = float(get_nsample()) * fraction_ones;

  // variance of the binomial distribution, JA98, Eqn. A6
  // cf. http://mathworld.wolfram.com/BinomialDistribution.html, Eqn. 14
  float nlo_variance = nlo_mean * (1.0 - fraction_ones);

  // the root mean square deviation
  float nlo_sigma = sqrt( nlo_variance );

  if (verbose)
    cerr << "  nlo_mean=" << nlo_mean << endl
         << "  nlo_sigma=" << nlo_sigma << endl;

  // backward compatibility
  if (psrdisp_compatible) {

    // in psrdisp, sigma was incorrectly set as
    nlo_sigma = sqrt( float(get_nsample()) );

    cerr << "dsp::TwoBitCorrection psrdisp compatibility\n"
      "   setting nlo_sigma to " << nlo_sigma << endl;

  }

  n_max = unsigned (nlo_mean + (cutoff_sigma * nlo_sigma));

  if (n_max >= get_nsample()) {
    if (verbose)
      cerr << "dsp::TwoBitCorrection::set_limits resetting nmax:"
	   << n_max << " to nsample-2:" << get_nsample()-1 << endl;
    n_max = get_nsample()-1;
  }

  if (cutoff_sigma * nlo_sigma >= nlo_mean+1.0) {
    if (verbose)
      cerr << "dsp::TwoBitCorrection::set_limits resetting nmin:"
	   << n_min << " to one:1" << endl;
    n_min = 1;
  }
  else 
    n_min = unsigned (nlo_mean - (cutoff_sigma * nlo_sigma));
  
  if (verbose) cerr << "dsp::TwoBitCorrection::set_limits nmin:"
		    << n_min << " and nmax:" << n_max << endl;
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

  if (verbose) cerr << "dsp::TwoBitCorrection::build"
		    << " nsamp=" << get_nsample()
		    << " cutoff=" << cutoff_sigma << "sigma\n";

  if (get_ndig()<1)
    throw Error (InvalidParam, "dsp::TwoBitCorrection::build",
		 "invalid number of digitizers=%d", get_ndig());

  if (!get_ndig_per_byte() || TwoBitTable::vals_per_byte % get_ndig_per_byte())
    throw Error (InvalidParam, "dsp::TwoBitCorrection::build",
		 "invalid channels_per_byte=%d", get_ndig_per_byte());
 
  if (!table)
    throw Error (InvalidParam, "dsp::TwoBitCorrection::build",
		 "no TwoBitTable");

  set_limits ();

  unsigned n_range = n_max - n_min + 1;

  bool huge = (get_ndig_per_byte() == 1);

  unsigned size = TwoBitTable::vals_per_byte;
  if (huge)
    size *= TwoBitTable::unique_bytes;

  if (verbose) cerr << "dsp::TwoBitCorrection::build allocate buffers\n";
  dls_lookup.resize (n_range * size);

  generate (&(dls_lookup[0]), 0, n_min, n_max, get_nsample(), table, huge);

  zero_histogram ();

  nlo_build ();

  built = true;

  if (verbose) cerr << "dsp::TwoBitCorrection::build exits\n";

}

void dsp::TwoBitCorrection::nlo_build ()
{
  if (verbose)
    cerr << "dsp::TwoBitCorrection::nlo_build" << endl;

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
}

void dsp::TwoBitCorrection::generate (float* dls, float* spc, 
				      unsigned n_min, unsigned n_max, unsigned n_tot,
				      TwoBitTable* table, bool huge)
{
  for (unsigned nlo=n_min; nlo <= n_max; nlo++)  {
  
    /* Refering to JA98, nlo is the number of samples between x2 and x4, 
       and p_in is the left-hand side of Eq.44 */
    float p_in = (float) nlo / (float) n_tot;

    float lo, hi, A;
    if (change_levels) {
      output_levels (p_in, lo, hi, A);

      table->set_lo_val (lo);
      table->set_hi_val (hi);
    }
    
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
    if (change_levels) {
      if (spc) {
        *spc = A;
        spc ++;
      }
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
    thresholds, x2 and x4, where -x2 = x4 are optimally set to
    threshold=0.9674 of the noise power, as in Table 1 of JA98.
    Apply t=threshold*sigma to Equation 45, 
    (or -xl=xh=threshold to Equation A2) to get: */
float dsp::TwoBitCorrection::get_fraction_low () const
{
  return erf (threshold / sqrt(2.0));
}


void dsp::TwoBitCorrection::unpack ()
{
  if (input->get_nbit() != 2)
    throw Error (InvalidParam, "dsp::TwoBitCorrection::unpack",
		 "input not 2-bit sampled");

  uint64 ndat = input->get_ndat();

  unsigned samples_per_byte = TwoBitTable::vals_per_byte / get_ndig_per_byte();

  if (ndat % samples_per_byte)
    throw Error (InvalidParam, "dsp::TwoBitCorrection::check_input",
		 "input ndat="I64" != %dn", ndat, samples_per_byte);
  
  if (input->get_state() != Signal::Nyquist && 
      input->get_state() != Signal::Analytic)
    throw Error (InvalidParam, "dsp::TwoBitCorrection::check_input",
		 "input is detected");

  if (ndat < get_nsample())
    throw Error (InvalidParam, "dsp::TwoBitCorrection::unpack",
		 "input ndat="UI64" < nsample=%d", ndat, get_nsample());

  const unsigned char* rawptr = input->get_rawptr();

  unsigned ndig = get_ndig ();

  // weights are used only if output is a WeightedTimeseries
  unsigned* weights = 0;
  uint64 nweights = 0;

  for (unsigned idig=0; idig<ndig; idig++) {

    unsigned ipol = get_output_ipol (idig);
    unsigned ichan = get_output_ichan (idig);
    unsigned input_offset = get_input_offset (idig);

#ifdef _DEBUG
    cerr << "idig=" << idig << " ichan=" << ichan << " ipol=" << ipol 
	 << " input offset=" << input_offset << endl;
#endif

    const unsigned char* from = rawptr + input_offset;

    float* into = output->get_datptr (ichan, ipol) + get_output_offset (idig);

#ifdef _DEBUG
    cerr << "dsp::TwoBitCorrection::unpack idig=" << idig << "/" << ndig
	 << " from=" << (void*)from << " to=" << into << endl;
#endif

    // if the output TimeSeries is a weighted output, use its weights array
    if (weighted_output) {
      weights = weighted_output -> get_weights (0, ipol);
      nweights = weighted_output -> get_nweights ();
    }

    dig_unpack (into, from, ndat, idig, weights, unsigned(nweights));
      
  }  // for each polarization

  output->seek (input->get_request_offset());
  output->set_ndat (input->get_request_ndat());

}

void dsp::TwoBitCorrection::dig_unpack (float* output_data,
					const unsigned char* input_data, 
					uint64 ndat,
					unsigned digitizer,
					unsigned* weights,
					unsigned nweights)
{
  unsigned ndig = get_ndig_per_byte();

  if (ndig != 1)
    throw Error (InvalidState, "dsp::TwoBitCorrection::dig_unpack",
		 "number of digitizers per byte = %d must be == 1", ndig);

  if (verbose){
    cerr << "dsp::TwoBitCorrection::dig_unpack out=" << output_data << endl;
    fprintf(stderr,"input_data=%p\n",input_data);
    fprintf(stderr,"ndat="UI64"\n",ndat);
    cerr <<   "\n   digitizer=" << digitizer << " weights=" << weights << endl;
    cerr <<   " nweights=" << nweights << endl;
  }

  // 4 floating-point samples per byte
  const unsigned samples_per_byte = TwoBitTable::vals_per_byte;

  // 4*256 floating-point samples for all unique bytes
  const unsigned lookup_block_size =
    samples_per_byte * TwoBitTable::unique_bytes;

  unsigned long* hist = 0;
  if (keep_histogram)
    hist = get_histogram (digitizer);

  unsigned long n_weights = (unsigned long) ceil (float(ndat)/float(get_nsample()));

  assert (n_weights*get_nsample() >= ndat);

  if (weights && n_weights > nweights)
    throw Error (InvalidParam, "dsp::TwoBitCorrection::dig_unpack",
		 "weights array size=%d < nweights=%d", nweights, n_weights);

  uint64 bytes_left = ndat / samples_per_byte;
  uint64 bytes_per_weight = get_nsample() / samples_per_byte;
  uint64 bytes = bytes_per_weight;
  unsigned bt;
  unsigned pt;

  unsigned input_incr = get_input_incr ();
  unsigned output_incr = get_output_incr ();

  float* section = 0;
  float* fourval = 0;

  for (unsigned long wt=0; wt<n_weights; wt++) {

    const unsigned char* input_data_ptr = input_data;

    if (bytes > bytes_left) {
      input_data_ptr -= (bytes_per_weight - bytes_left) * input_incr;
      bytes = bytes_left;
    }

    // calculate the weight based on the last nsample pts
    unsigned n_lo = 0;
    for (bt=0; bt<bytes_per_weight; bt++) {
      n_lo += nlo_lookup [*input_data_ptr];
      input_data_ptr += input_incr;
    }

    if (hist)
      hist [n_lo] ++;

    // test if the number of low voltage states is outside the
    // acceptable limit or if this section of data has been previously
    // flagged bad (for example, due to bad data in the other polarization)
    if ( n_lo<n_min || n_lo>n_max || (weights && weights[wt] == 0) ) {

#ifdef _DEBUG
      cerr << "w[" << wt << "]=0 ";
#endif
      
      if (weights)
        weights[wt] = 0;
      
      // reduce the risk of other functions accessing un-initialized 
      // segments of the array
      for (bt=0; bt<bytes*samples_per_byte; bt++) {
	*output_data = 0.0;
	output_data += output_incr;
      }
      
    }
    
    else {
      
      input_data_ptr = input_data;
      section = &(dls_lookup[0]) + (n_lo-n_min) * lookup_block_size;
      
      for (bt=0; bt<bytes; bt++) {
	fourval = section + unsigned(*input_data_ptr) * samples_per_byte;
	for (pt=0; pt<samples_per_byte; pt++) {
	  *output_data = fourval[pt];
	  output_data += output_incr;
	}
	input_data_ptr += input_incr;
      }

      if (weights)
	weights[wt] = n_lo;
      
    }
    
    bytes_left -= bytes;
    input_data += bytes * input_incr;
  }
  
}
  
  
int64 dsp::TwoBitCorrection::stats(vector<double>& m, vector<double>& p)
{
  static float* lu_sum = 0;
  static float* lu_sumsq = 0;
  static unsigned char* lu_nlo = 0;

  if (!input)
    throw Error (InvalidState, "dsp::TwoBitCorrection::stats", "no input");

  if (input->get_nbit() != 2)
    throw Error (InvalidParam, "dsp::TwoBitCorrection::stats",
		 "input nbit != 2");

  if (input->get_state() != Signal::Nyquist)
    throw Error (InvalidParam, "dsp::TwoBitCorrection::stats",
		 "input state != Nyquist");

  if (lu_sum == NULL) {

    //if (verbose)
    cerr << "dsp::TwoBitCorrection::stats generate lookup table" <<endl;

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
  uint64 nweights = input->get_ndat() / get_nsample();

  unsigned npol = input->get_npol ();

  uint64 nbytes = input->get_nbytes() / npol;

  // number of bytes per weight sample
  uint64 nbytes_per_weight = nbytes / nweights;


  if (verbose)
    cerr << "dsp::TwoBitCorrection::stats npol=" << npol 
	 << " nweights=" << nweights << " bytespw=" << nbytes_per_weight
	 << endl;

  m.resize(npol);
  p.resize(npol);

  unsigned total_samples = 0;

  for (unsigned ipol=0; ipol < npol; ipol++) {

    unsigned long* hist = get_histogram(ipol);

    double sum = 0;
    double sumsq = 0;
    total_samples = 0;

    const unsigned char* data = input->get_rawptr() + ipol;
    unsigned char datum;

    for (uint64 iwt=0; iwt<nweights; iwt++) {

      unsigned long nlo = 0;

      for (uint64 ipwt=0; ipwt < nbytes_per_weight; ipwt++) {

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
    cerr << "dsp::TwoBitCorrection::stats return total samples " 
	 << total_samples <<  endl;

  // return the total number of timesamples measured
  return total_samples;
}


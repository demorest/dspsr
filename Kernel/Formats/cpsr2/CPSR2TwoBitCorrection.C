#include "CPSR2TwoBitCorrection.h"
#include "TwoBitTable.h"
#include "Timeseries.h"

#include "string_utils.h"
#include "genutil.h"

//! Null constructor
dsp::CPSR2TwoBitCorrection::CPSR2TwoBitCorrection (unsigned _nsample,
						   float _cutoff_sigma)
  : TwoBitCorrection ("CPSR2TwoBitCorrection", outofplace)
{
  if (!_nsample)
    throw stringprintf ("CPSR2TwoBitCorrection:: invalid nsample %d",
			_nsample);

  if (_cutoff_sigma <= 0)
    throw stringprintf ("CPSR2TwoBitCorrection:: invalid cut_off sigma %f",
			_cutoff_sigma);

  nsample = _nsample;
  cutoff_sigma = _cutoff_sigma;
  
  nchannel = 2;
  type = TwoBitTable::OffsetBinary;
  channels_per_byte = 1;

}

void dsp::CPSR2TwoBitCorrection::unpack ()
{
  if (input->get_state() != Observation::Nyquist)
    throw_str ("CPSR2TwoBitCorrection::unpack input not real sampled");

  if (input->get_nbit() != 2)
    throw_str ("CPSR2TwoBitCorrection::unpack input not 2-bit sampled");

  int64 ndat = input->get_ndat();

  if (ndat % 4)
    throw_str ("CPSR2TwoBitCorrection::unpack input ndat="I64" != 4n", ndat);
  
  if (ndat < nsample)
    throw_str ("CPSR2TwoBitCorrection::unpack input ndat="I64" < nsample=%d",
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

void dsp::CPSR2TwoBitCorrection::poln_unpack (float* data,
					      const unsigned char* raw, 
					      uint64 ndat,
					      unsigned long* hist,
					      unsigned gap)
{
  static unsigned char* lu_nlo = 0;

  if (!lu_nlo) {
    lu_nlo = new unsigned char [256];
    TwoBitTable* table = new TwoBitTable (TwoBitTable::OffsetBinary);

    const float* fourvals = 0;
    float lo_valsq = table->get_lo_val() * table->get_lo_val();

    for (unsigned byte = 0; byte < 256; byte++) {

      lu_nlo[byte] = 0;
      fourvals = table->get_four_vals (byte);

      for (unsigned ifv=0; ifv<4; ifv++)
	if (fourvals[ifv]*fourvals[ifv] == lo_valsq)
	  lu_nlo[byte] ++;

    }
    delete table;

  }

  unsigned long n_weights = (unsigned long) ceil (float(ndat)/float(nsample));

  assert (n_weights*nsample >= ndat);

  // four two-bit samples per byte in this unpacking scheme
  const unsigned samples_per_byte = 4;

  // four floating-point samples for each of 256 possible bytes in each block
  const unsigned lookup_block_size = samples_per_byte * 256;

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
      n_lo += lu_nlo [*rawptr];
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


int64 dsp::CPSR2TwoBitCorrection::stats(vector<double>& m, vector<double>& p)
{
  static float* lu_sum = 0;
  static float* lu_sumsq = 0;
  static unsigned char* lu_nlo = 0;

  if (!input)
    throw_str ("CPSR2TwoBitCorrection::stats no input");

  if (input->get_nbit() != 2)
    throw_str ("CPSR2TwoBitCorrection::stats input nbit != 2");

  if (input->get_state() != Observation::Nyquist)
    throw_str ("CPSR2TwoBitCorrection::stats input state != Nyquist");

  if (int(histograms.size()) != nchannel) {
    histograms.resize(nchannel);
    for (int ichan=0; ichan < nchannel; ichan++)
      histograms[ichan].resize (nsample);
  }

  if (lu_sum == NULL) {

    //if (verbose)
    cerr << "CPSR2TwoBitCorrection::stats generate lookup table" <<endl;

    TwoBitTable* table = new TwoBitTable (TwoBitTable::OffsetBinary);

    // calculate the sum and sum-squared of the four time samples in each byte
    lu_sum = new float [256];
    lu_sumsq = new float [256];
    lu_nlo = new unsigned char [256];

    const float* fourvals = 0;
    float val = 0;
    float valsq = 0;
    float lo_valsq = table->get_lo_val() * table->get_lo_val();

    for (unsigned byte = 0; byte < 256; byte++) {
      
      lu_sum[byte] = 0;
      lu_sumsq[byte] = 0;
      lu_nlo[byte] = 0;

      fourvals = table->get_four_vals (byte);

      for (unsigned ifv=0; ifv<4; ifv++) {
	val = fourvals[ifv];
	valsq = val * val;

	lu_sum[byte] += val;
	lu_sumsq[byte] += valsq;

	if (valsq == lo_valsq)
	  lu_nlo[byte] ++;
      }
    }

    delete table;
  }

  // number of weight samples
  unsigned nweights = input->get_ndat() / nsample;

  unsigned npol = input->get_npol ();

  unsigned nbytes = input->nbytes() / npol;

  // number of bytes per weight sample
  unsigned nbytes_per_weight = nbytes / nweights;


  if (verbose)
    cerr << "CPSR2TwoBitCorrection::stats npol=" << npol 
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
	total_samples += 4;
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
    cerr << "CPSR2TwoBitCorrection::stats return total samples " 
	 << total_samples <<  endl;

  // return the total number of timesamples measured
  return total_samples;
}


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
  table = 0;
}

void dsp::CPSR2TwoBitCorrection::unpack ()
{
  if (input->get_ndim() != 1)
    throw_str ("CPSR2TwoBitCorrection::operation input not real sampled");

#if 0
  int64 ndat = input->get_ndat();
  const unsigned char* rawptr = input->get_rawptr();

  for (int ipol=0; ipol<input->get_npol(); ipol++) {



  }  // for each polarization
#endif

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

  if (!table)
    table = new TwoBitTable (TwoBitTable::OffsetBinary);

  if (lu_sum == NULL) {

    //if (verbose)
    cerr << "CPSR2TwoBitCorrection::stats generate lookup table" <<endl;

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

    unsigned long* hist = histograms + nsample * ipol;
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

void dsp::CPSR2TwoBitCorrection::build (int nsamp, float sigma)
{
  // setup the lookup table
  TwoBitCorrection::build (nsamp, sigma, TwoBitTable::OffsetBinary, true);
}

void dsp::CPSR2TwoBitCorrection::destroy ()
{

}


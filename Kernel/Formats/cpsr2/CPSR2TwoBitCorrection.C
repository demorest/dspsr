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
}

void dsp::CPSR2TwoBitCorrection::unpack ()
{
  if (input->get_ndim() != 1)
    throw_str ("TwoBitCorrection::operation input not real sampled");

  int64 ndat = input->get_ndat();
  const unsigned char* rawptr = input->get_rawptr();

  for (int ipol=0; ipol<input->get_npol(); ipol++) {



  }  // for each polarization

}

int64 dsp::CPSR2TwoBitCorrection::stats(vector<double>& m, vector<double>& p)
{
  static float* lu_sum = 0;
  static float* lu_sumsq = 0;
  static unsigned char* lu_nlo = 0;

  if (!input)
    throw string ("dsp::CPSR2TwoBitCorrection::stats no input");

  if (input->get_nbit() != 2)
    throw string ("dsp::CPSR2TwoBitCorrection::stats input nbit != 2");

  if (input->get_state() != Observation::Nyquist)
    throw string ("dsp::CPSR2TwoBitCorrection::stats input state != Nyquist");

  if (!table)
    table = new TwoBitTable (TwoBitTable::OffsetBinary);

  if (!lu_sum) {
    // calculate the sum and sum-squared of the four time samples in each byte
    lu_sum = new float [256];
    lu_sumsq = new float [256];
    lu_nlo = new unsigned char [256];

    const float* fourvals = 0;
    float val = 0;
    float valsq = 0;
    float lo_valsq = table->get_lo_val() * table->get_lo_val();

    for (unsigned char byte = 0; byte < 256; byte++) {

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

  // number of bytes per weight sample
  unsigned nbytes_per_weight = input->nbytes () / nweights;

  unsigned npol = input->get_npol ();

  for (unsigned ipol=0; ipol < npol; ipol++) {

    unsigned long* hist = histograms + nsample * ipol;
    double sum = 0;
    double sumsq = 0;

    const unsigned char* data = input->get_rawptr();
    unsigned char datum;

    for (unsigned iwt=0; iwt<nweights; iwt++) {

      unsigned long nlo = 0;

      for (unsigned ipwt=0; ipwt < nbytes_per_weight; ipwt++) {

	datum = *data;
	data += ipol;

	sum += lu_sum[datum];
	sumsq += lu_sumsq[datum];
	nlo += lu_nlo[datum];

      }

      hist[nlo] ++;
    }

  }

  unsigned nbytes_total = nweights * nbytes_per_weight;
  unsigned nsamples_per_byte = unsigned (1.0/input->nbyte());

  // return the total number of timesamples measured
  return nbytes_total * nsamples_per_byte; 
}

void dsp::CPSR2TwoBitCorrection::build (int nsamp, float sigma)
{
  // setup the lookup table
  TwoBitCorrection::build (nsamp, sigma, TwoBitTable::OffsetBinary, true);
}

void dsp::CPSR2TwoBitCorrection::destroy ()
{

}


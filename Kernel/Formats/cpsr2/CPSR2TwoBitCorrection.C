#include "CPSR2TwoBitCorrection.h"
#include "Timeseries.h"
#include "genutil.h"

//! Null constructor
dsp::CPSR2TwoBitCorrection::CPSR2TwoBitCorrection (int _nsample,
						   float _cutoff_sigma)
  : TwoBitCorrection ("CPSR2TwoBitCorrection", outofplace)
{
  nchannel = 4;
  build (_nsample, _cutoff_sigma);
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


void dsp::CPSR2TwoBitCorrection::build (int nsamp, float sigma)
{
  // delete the old space
  CPSR2TwoBitCorrection::destroy();

  // setup the lookup table
  TwoBitCorrection::build (nsamp, sigma);

}

void dsp::CPSR2TwoBitCorrection::destroy ()
{

}


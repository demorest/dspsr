#include "dsp/SMROTwoBitCorrection.h"
#include "dsp/TwoBitTable.h"

bool dsp::SMROTwoBitCorrection::matches (const Observation* observation)
{
  return observation->get_machine() == "SMRO"
    && observation->get_nbit() == 2
    && observation->get_state() == Signal::Nyquist;
}

//! Null constructor
dsp::SMROTwoBitCorrection::SMROTwoBitCorrection ()
  : SubByteTwoBitCorrection ("SMROTwoBitCorrection")
{
  //threshold = 1.5;
  //table = new TwoBitTable (TwoBitTable::SignMagnitude);
  table = new TwoBitTable (TwoBitTable::OffsetBinary);
  table->set_flip(true);
}

/*! SMRO has four digitizers: Potentially 4 channels */
unsigned dsp::SMROTwoBitCorrection::get_ndig () const
{
  return 4;
}

/*! Each 2-bit sample from each digitizer is packed into one byte */
unsigned dsp::SMROTwoBitCorrection::get_ndig_per_byte () const
{ 
  return 4;
}

/*! Override the dig_unpack to change the number of polns */
/*void dsp::SMROTwoBitCorrection::dig_unpack (float* output_data,
					const unsigned char* input_data, 
					uint64 ndat,
					unsigned digitizer,
					unsigned* weights,
					unsigned nweights)
{
  dsp::SubByteTwoBitCorrection::dig_unpack(output_data, input_data, ndat, digitizer, weights, nweights);
  
  cout << "yes" << endl;
  observation->set_nchan(2);
  observation->set_npol(2);
  
}
*/

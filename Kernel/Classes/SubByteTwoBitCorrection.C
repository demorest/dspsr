#include "dsp/SubByteTwoBitCorrection.h"
#include "dsp/SubByte_dig_unpack.h"

#include "dsp/TwoBitMask.h"

#include <iostream>

// #define _DEBUG 1

dsp::SubByteTwoBitCorrection::SubByteTwoBitCorrection (const char* name)
  : TwoBitCorrection (name)
{
  values = 0;
}

dsp::SubByteTwoBitCorrection::~SubByteTwoBitCorrection ()
{
  destroy ();
}

/*! By default, both polarizations are output in one byte */
unsigned dsp::SubByteTwoBitCorrection::get_ndig_per_byte () const
{ 
  return 2;
}

/*! By default, the data is not interleaved byte by byte */
unsigned dsp::SubByteTwoBitCorrection::get_input_offset (unsigned idig) const
{
  return 0;
}

/*! By default, the data is not interleaved byte by byte */
unsigned dsp::SubByteTwoBitCorrection::get_input_incr () const 
{
  return 1;
}

/*! By default, MSB y1 x1 y0 x0 LSB */
unsigned
dsp::SubByteTwoBitCorrection::get_shift (unsigned idig, unsigned samp) const
{
  return (idig + samp * 2) * 2;
}

/* By default, there may be one time sample from each of two or four
   digitizer outputs in each byte.  For an example of code with two
   samples from each of two digitizers, with bits ordered in a
   different way, please see mark5/Mark5TwoBitCorrection.C. */
void dsp::SubByteTwoBitCorrection::dig_unpack (float* outptr,
					       const unsigned char* inptr,
					       uint64 ndat,
					       unsigned digitizer,
					       unsigned* weights,
					       unsigned nweights)
{
  ShiftMask<2> mask;
  mask.shift[0] = get_shift(digitizer,0);
  if (get_ndig_per_byte() == 2)
    mask.shift[1] = get_shift(digitizer,1);

  dig_unpack (mask, outptr, inptr, ndat, digitizer, weights, nweights);
}

void dsp::SubByteTwoBitCorrection::build ()
{
  if (verbose)
    cerr << "dsp::SubByteTwoBitCorrection::build" << endl;

  // delete the old space
  SubByteTwoBitCorrection::destroy();

  // setup the lookup table
  TwoBitCorrection::build ();

  // create the new space
  values = new unsigned char [get_nsample()];
}

void dsp::SubByteTwoBitCorrection::nlo_build ()
{
  if (verbose)
    cerr << "dsp::SubByteTwoBitCorrection::nlo_build" << endl;

  float fourvals [TwoBitTable::vals_per_byte];
  float lo_valsq = 1.0;

  // flatten the table again (precision errors cause mismatch of lo_valsq)
  table->set_lo_val (1.0);
  table->four_vals (fourvals);

  for (unsigned ifv=0; ifv<TwoBitTable::vals_per_byte; ifv++)
    if (fourvals[ifv]*fourvals[ifv] == lo_valsq)
      lovoltage[ifv] = 1;
    else
      lovoltage[ifv] = 0;
}

void dsp::SubByteTwoBitCorrection::destroy ()
{
  if (values != NULL) delete [] values;  values = NULL;
}


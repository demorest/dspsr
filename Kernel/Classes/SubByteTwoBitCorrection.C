/***************************************************************************
 *
 *   Copyright (C) 2003 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SubByteTwoBitCorrection.h"
#include "dsp/excision_unpack.h"
#include "dsp/StepIterator.h"
#include <iostream>

using namespace std;

// #define _DEBUG 1

dsp::SubByteTwoBitCorrection::SubByteTwoBitCorrection (const char* name)
  : TwoBitCorrection (name)
{
}

dsp::SubByteTwoBitCorrection::~SubByteTwoBitCorrection ()
{
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
void dsp::SubByteTwoBitCorrection::dig_unpack (const unsigned char* input_data,
					       float* output_data,
					       uint64 nfloat,
					       unsigned long* hist,
					       unsigned* weights,
					       unsigned nweights)
{
  StepIterator<const unsigned char> iterator (input_data);
  iterator.set_increment ( get_input_incr() );

  unpacker.mask.shift[0] = get_shift (current_digitizer, 0);

  ExcisionUnpacker::excision_unpack (unpacker, iterator,
				     output_data, nfloat,
                                     hist, weights, nweights);
}

void dsp::SubByteTwoBitCorrection::build ()
{
  //if (verbose)
    cerr << "dsp::SubByteTwoBitCorrection::build" << endl;

  ExcisionUnpacker::build ();

  unpacker.set_nlow_min (nlow_min);
  unpacker.set_nlow_max (nlow_max);

  cerr << "dsp::SubByteTwoBitCorrection::build ndat=" << get_ndat_per_weight() << " ndim=" << get_ndim_per_digitizer() << endl;
  unpacker.lookup_build (get_ndat_per_weight(), 
                         get_ndim_per_digitizer(),
                         table, &ja98);

  unpacker.nlow_build (table);
}


#include <stdio.h>
#include <iostream>
#include "TwoBitTable.h"

dsp::TwoBitTable::TwoBitTable (Type _type)
{
  // 256 possible unsigned chars * 4 floats per byte
  table = 0;
  built = false;

  lo_val = 0.25;
  hi_val = 0.75;

  type = _type;
}


dsp::TwoBitTable::~TwoBitTable ()
{
  if (table) delete [] table; table = 0;
}

//! Set the value of the low voltage state
void dsp::TwoBitTable::set_lo_val (float _lo_val)
{
  if (lo_val != _lo_val)
    built = false;

  lo_val = _lo_val;
}

//! Set the value of the high voltage state
void dsp::TwoBitTable::set_hi_val (float _hi_val)
{
  if (hi_val != _hi_val)
    built = false;

  hi_val = _hi_val;
}

//! Set the digitization convention
void dsp::TwoBitTable::set_type (Type _type)
{
  if (type != _type)
    built = false;

  type = _type;
}

void dsp::TwoBitTable::build ()
{
  if (built)
    return;

  if (!table)
    table = new float [ 256 * 4 ];

  generate (table);
}

const float* dsp::TwoBitTable::get_four_vals (unsigned byte)
{
  if (!built)
    build ();

  return table + 4 * byte; 
}


/*!
  \param table pointer to space for at least 4*256=1024 floating point values,
  representing the four outputs for each of 256 possible bytes.
*/
void dsp::TwoBitTable::generate (float* table) const
{
  float voltages[4];
  four_vals (voltages);

  float* tabval = table;

  for (unsigned byte=0; byte<256; byte++) {
    for (unsigned sample=0; sample<4; sample++) {
      *tabval = voltages[twobit (byte, sample)];
      tabval ++;
    }
  }
}

/*!  By default, each 8-bit byte is treated as four consecutive 2-bit
  time samples, with the first time sample in the most significant two bits.

  \param byte the byte pattern containing four time samples
  \param sample the sample to extract from the byte */
unsigned dsp::TwoBitTable::twobit (unsigned byte, unsigned sample) const
{
  unsigned char mask  = 0x03;
  unsigned char shift = 6 - sample*2;

  return (byte>>shift) & mask;
}

void dsp::TwoBitTable::four_vals (float* vals) const
{
  switch (type) {

  case OffsetBinary:
    vals[0] = -hi_val;
    vals[1] = -lo_val;
    vals[2] = lo_val;
    vals[3] = hi_val;
    break;
    
  case SignMagnitude:
    vals[0] = lo_val;
    vals[1] = hi_val;
    vals[2] = -lo_val;
    vals[3] = -hi_val;
    break;
    
  case TwosComplement:
    vals[0] = lo_val;
    vals[1] = hi_val;
    vals[2] = -hi_val;
    vals[3] = -lo_val;
    break;

  default:
    throw "TwoBitTable::four_vals unsupported type";
  }
}

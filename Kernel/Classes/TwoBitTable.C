/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/TwoBitTable.h"

//! Number of unique 8-bit combinations
const unsigned dsp::TwoBitTable::unique_bytes = 1<<8; // (256)

//! Number of 2-bit values per byte
const unsigned dsp::TwoBitTable::vals_per_byte = 4;

dsp::TwoBitTable::TwoBitTable (Type _type)
{
  table = 0;
  built = false;

  lo_val = 0.25;
  hi_val = 0.75;

  type = _type;

  flip = false;
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

void dsp::TwoBitTable::set_flip (bool flipped)
{
  flip = flipped;
}


void dsp::TwoBitTable::build ()
{
  if (built)
    return;

  if (!table)
    table = new float [ unique_bytes * vals_per_byte ];

  generate (table);
}

const float* dsp::TwoBitTable::get_four_vals (unsigned byte)
{
  if (!built)
    build ();

  return table + vals_per_byte * byte; 
}


/*!
  \param table pointer to space for at least 4*256=1024 floating point values,
  representing the four outputs for each of 256 possible bytes.
*/
void dsp::TwoBitTable::generate (float* table) const
{
  float voltages[vals_per_byte];
  four_vals (voltages);

  float* tabval = table;

  for (unsigned byte=0; byte<unique_bytes; byte++) {
    for (unsigned val=0; val<vals_per_byte; val++) {
      *tabval = voltages[twobit (byte, val)];
      tabval ++;
    }
  }
}

/*! Each 8-bit byte is treated as four 2-bit numbers, ordered from 0 to 3.
  By default, the first 2-bit number (val==0) is in the most significant
  two bits and the last 2-bit number (val==3) is in the least significant
  two bits.

  \param byte the byte pattern containing four 2-bit numbers
  \param val the value to extract from the byte */
unsigned dsp::TwoBitTable::twobit (unsigned byte, unsigned val) const
{
  unsigned char mask  = 0x03;
  unsigned char shift = 6 - val*2;

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
  
  // Swap the order of the bits, e.g. SignMag becomes MagSign
  if(flip){
    float tmp;
    tmp = vals[1];
    vals[1] = vals[2];
    vals[2] = tmp;
  }
    
}

/***************************************************************************
 *
 *   Copyright (C) 2008 Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/BitTable.h"

using namespace std;

const unsigned dsp::BitTable::bits_per_byte = 8;

const unsigned dsp::BitTable::unique_bytes = 1 << (bits_per_byte-1);

unsigned build_mask (unsigned nbit)
{
  unsigned mask = 0;

  for (unsigned bit = 1; bit <= nbit; bit << 1)
    mask |= bit;

  return mask;
}

dsp::BitTable::BitTable (unsigned _nbit, Type _type, bool _build)
  :
  type( _type ),
  nbit( _nbit ),
  values_per_byte( bits_per_byte / nbit ),
  unique_values( 1 << (nbit-1) ),
  nbit_mask( build_mask(nbit) )
{
  cerr << "unique bytes=" << unique_bytes << " values=" << unique_values
       << " values per byte=" << values_per_byte << endl;

  table = 0;

  if (_build)
    build ();
}

dsp::BitTable::~BitTable ()
{
  if (table) delete [] table; table = 0;
}

const float* dsp::BitTable::get_values (unsigned byte)
{
  return table + values_per_byte * byte; 
}

void dsp::BitTable::build ()
{
  if (!table)
    table = new float [ unique_bytes * values_per_byte ];

  generate (table);
}

/*!
  \param table pointer to space for at least unique_bytes * values_per_byte 
         floating point values
*/
void dsp::BitTable::generate (float* table) const
{
  float values[unique_values];
  generate_unique_values (values);

  float* tabval = table;

  for (unsigned byte=0; byte<unique_bytes; byte++)
  {
    for (unsigned val=0; val<values_per_byte; val++)
    {
      *tabval = values[extract (byte, val)];
      tabval ++;
    }
  }
}

/*! Each byte is treated as unique_values consecutive values, from
  most significant bit to least significant bit

  \param byte the byte pattern containing unique_values values
  \param i the index of the value to extract from the byte */
unsigned dsp::BitTable::extract (unsigned byte, unsigned i) const
{
  unsigned shift = bits_per_byte - nbit * (i+1);

  return (byte>>shift) & nbit_mask;
}

void dsp::BitTable::generate_unique_values (float* values) const
{
  float spacing = 1.0 / float(unique_values);
  float middle = float(unique_values - 1) / 2.0;

  switch (type) {

  case OffsetBinary:
    for (unsigned i=0; i<unique_values; i++)
      values[i] = (float(i) - middle) * spacing;
    break;

  case TwosComplement:
  {
    unsigned half = unique_values / 2;
    for (unsigned i=0; i<unique_values; i++)
      values[i] = (float((i+half)%unique_values) - middle) * spacing;
    break;
  }

  default:
    throw Error (InvalidState, "BitTable::generate_unique_values",
		 "unsupported type");
  }
   
}

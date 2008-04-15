/***************************************************************************
 *
 *   Copyright (C) 2008 Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/BitTable.h"

using namespace std;

const unsigned dsp::BitTable::bits_per_byte = 8;

const unsigned dsp::BitTable::unique_bytes = 1 << bits_per_byte;

unsigned build_mask (unsigned nbit)
{
  unsigned mask = 0;
  unsigned add  = 1;

  for (unsigned bit = 0; bit < nbit; bit ++)
  {
    mask |= add;
    add <<= 1;
  }

#ifdef _DEBUG
  cerr << "build_mask: nbit=" << nbit << " mask=" << mask << endl;
#endif

  return mask;
}

dsp::BitTable::BitTable (unsigned _nbit, Type _type, bool _reverse)
  :
  type( _type ),
  nbit( _nbit ),
  reverse_bits( _reverse ),
  values_per_byte( bits_per_byte / nbit ),
  unique_values( 1 << nbit ),
  nbit_mask( build_mask(nbit) )
{

#ifdef _DEBUG
  cerr << "unique bytes=" << unique_bytes << " values=" << unique_values
       << " values per byte=" << values_per_byte << endl;
#endif

  table = 0;
}

dsp::BitTable::~BitTable ()
{
  if (table) delete [] table; table = 0;
}

const float* dsp::BitTable::get_values (unsigned byte)
{
  if (!table)
    build ();

  return table + values_per_byte * byte; 
}

void dsp::BitTable::build ()
{
  if (!table)
    table = new float [ unique_bytes * values_per_byte ];

  generate (table);
}

//! Reverses the order of bits
template<typename T>
T reverse (T value, unsigned N)
{
  T one = 1;
  T result = 0;
  for (unsigned bit=0; bit < N; bit++)
    if ((value >> bit) & one)
      result |= one << (N-bit-1);
  return result;
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
    if (reverse_bits)
    {
      cerr << byte << " -> ";
      byte = reverse (byte, 8);
      cerr << byte << endl;
    }

    for (unsigned val=0; val<values_per_byte; val++)
    {
      unsigned sample = extract (byte, val);
#ifdef _DEBUG
      cerr << "byte=" << byte << " val=" << val << " sample=" << sample << endl;
#endif
      *tabval = values[sample];
      tabval ++;
    }
  }
}

/*! Each byte is treated as unique_values consecutive values, from
  most significant bit to least significant bit

  \param byte the byte pattern containing unique_values values
  \param i the index of the value to extract from the byte
*/
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


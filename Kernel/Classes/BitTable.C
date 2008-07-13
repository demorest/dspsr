/***************************************************************************
 *
 *   Copyright (C) 2008 Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/BitTable.h"
#include "JenetAnderson98.h"
#include "NormalDistribution.h"

#include <math.h>
#include <assert.h>

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

const float* dsp::BitTable::get_values (unsigned byte) const
{
  if (!table)
    const_cast<BitTable*>(this)->build ();

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

  for (unsigned b=0; b<unique_bytes; b++)
  {
    unsigned byte = b;

    if (reverse_bits)
    {
      // cerr << byte << " -> ";
      byte = reverse (byte, 8);
      // cerr << byte << endl;
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
  double output_spacing = 1.0 / double(unique_values);
  double output_middle = double(unique_values - 1) / 2.0;

  unsigned input_middle = unique_values / 2;
  double input_spacing = JenetAnderson98::get_optimal_spacing (nbit);

#if _DEBUG
  cerr << "optimal input spacing = " << input_spacing << endl;
  cerr << "last level = " << input_spacing * (unique_values/2-1) << endl;
#endif

  unsigned input_offset = 0;
  if (type == TwosComplement)
    input_offset = unique_values / 2;

  NormalDistribution normal;
  double cumulative_probability = 0.0;
  double variance = 0.0;

  for (unsigned i=0; i<unique_values; i++)
  {
    double output = (double(i) - output_middle) * output_spacing;
    values[(i+input_offset)%unique_values] = output;

    if (i < input_middle)
    {
      double threshold = double(int(i+1) - int(input_middle)) * input_spacing;
      double cumulative = normal.cumulative_distribution (threshold);
      double interval = cumulative - cumulative_probability;
      cumulative_probability = cumulative;

      variance += output*output * interval;

#ifdef _DEBUG
      cerr << i << " t=" << threshold << " c=" << interval 
           << " v=" << variance << endl;
#endif

    }
  }

  assert (cumulative_probability == 0.5);
  variance *= 2.0;

  // scale such that the variance is unity
  double scale = 1.0/sqrt(variance);
  for (unsigned i=0; i<unique_values; i++)
  {
    // cerr << i << " " << values[i];
    values[i] *= scale;
    // cerr << " " << values[i] << endl;
  }
}

double dsp::BitTable::get_optimal_variance () const
{
  return 1.0;
}

/*
  return the sampling threshold nearest to and less than unity
*/
double dsp::BitTable::get_nlow_threshold () const
{
  double input_spacing = JenetAnderson98::get_optimal_spacing (nbit);
  unsigned steps = unsigned (1.0 / input_spacing);
  return steps * input_spacing;
}

void dsp::BitTable::get_nlow_lookup (char* nlow_lookup) const
{
  double nlow_threshold = get_nlow_threshold ();

  const float* lookup = get_values ();

  for (unsigned ichar=0; ichar < unique_bytes; ichar++)
  {
    nlow_lookup[ichar] = 0;

    for (unsigned val=0; val<values_per_byte; val++)
    {
      if (fabs(*lookup) < nlow_threshold)
	nlow_lookup[ichar] ++;
      lookup ++;
    }
  }
}

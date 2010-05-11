/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/TwoBitFour.h"

// 4 floating-point samples per byte
const unsigned dsp::TwoBitFour::samples_per_byte = 4;

// 4 floating-point samples per byte times 256 unique bytes
const unsigned dsp::TwoBitFour::lookup_block_size = 4 * 256;

//! Build the output value lookup table
void dsp::TwoBitFour::lookup_build (TwoBitTable* table, JenetAnderson98* ja98)
{
  TwoBitLookup::lookup_build (table, ja98);
  nlow_build (table);
}

void dsp::TwoBitFour::nlow_build (TwoBitTable* table)
{
  table->set_lo_val (1.0);
  table->rebuild();

  float lo_valsq = 1.0;

  for (unsigned byte = 0; byte < BitTable::unique_bytes; byte++)
  {
    nlow_lookup[byte] = 0;
    const float* fourvals = table->get_values (byte);

    for (unsigned ifv=0; ifv<table->get_values_per_byte(); ifv++)
      if (fourvals[ifv]*fourvals[ifv] == lo_valsq)
	nlow_lookup[byte] ++;
  }
}

void dsp::TwoBitFour::get_lookup_block (float* lookup, TwoBitTable* table)
{
  table->rebuild();

  /* Generate the 256 sets of four output floating point values
     corresponding to each byte */
  table->generate ( lookup );
}

unsigned dsp::TwoBitFour::get_lookup_block_size ()
{
  return lookup_block_size;
}

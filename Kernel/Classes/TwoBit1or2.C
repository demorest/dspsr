/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/TwoBit1or2.h"
#include "JenetAnderson98.h"

#include <assert.h>

dsp::TwoBit1or2::TwoBit1or2 ()
{
}

dsp::TwoBit1or2::~TwoBit1or2 ()
{
}

//! Build the output value lookup table
void dsp::TwoBit1or2::lookup_build (TwoBitTable* table, JenetAnderson98* ja98)
{
  nlow_build (table);
  TwoBitLookup::lookup_build (table, ja98);
}

void dsp::TwoBit1or2::nlow_build (TwoBitTable* table)
{
  float fourvals [4];
  float lo_valsq = 1.0;

  // flatten the table again (precision errors cause mismatch of lo_valsq)
  table->set_lo_val (1.0);
  table->generate_unique_values (fourvals);

  for (unsigned ifv=0; ifv<4; ifv++)
    if (fourvals[ifv]*fourvals[ifv] == lo_valsq)
      nlow_lookup[ifv] = 1;
    else
      nlow_lookup[ifv] = 0;
}

void dsp::TwoBit1or2::get_lookup_block (float* lookup, TwoBitTable* table)
{
  // Generate the four output levels corresponding to each 2-bit number
  table->generate_unique_values ( lookup );
}

unsigned dsp::TwoBit1or2::get_lookup_block_size ()
{
  // four unique 2-bit numbers
  return 4;
}

void dsp::TwoBit1or2::create ()
{
  TwoBitLookup::create ();
  lookup_base = new float [(nlow_max - nlow_min + 1) * 4];
}


/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/TwoBitFour.h"
#include "JenetAnderson98.h"

// 4 floating-point samples per byte
const unsigned dsp::TwoBitFour::samples_per_byte = 4;

// 4 floating-point samples per byte times 256 unique bytes
const unsigned dsp::TwoBitFour::lookup_block_size = 4 * 256;

dsp::TwoBitFour::TwoBitFour ()
{
  nlow = nlow_min = nlow_max = 0;
  lookup_base = 0;
}

dsp::TwoBitFour::~TwoBitFour ()
{
  destroy ();
}

void dsp::TwoBitFour::destroy ()
{
  if (lookup_base)
    delete [] lookup_base;

  lookup_base = 0;
}

void dsp::TwoBitFour::set_nlow_min (unsigned min)
{
  nlow_min = min;
  destroy ();
}

void dsp::TwoBitFour::set_nlow_max (unsigned max)
{
  nlow_max = max;
  destroy ();
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

void dsp::TwoBitFour::lookup_build (unsigned nsamp,
				    TwoBitTable* table,
				    JenetAnderson98* ja98)
{
  destroy ();

  assert (nlow_max > nlow_min);
  assert (nsamp > nlow_max);

  lookup_base = new float [(nlow_max - nlow_min + 1) * lookup_block_size];

  float* lookup = lookup_base;

  for (unsigned nlo = nlow_min; nlo <= nlow_max; nlo++)
  {
    /* Refering to JA98, nlo is the number of samples between x2 and x4, 
       and p_in is the left-hand side of Eq.44 */

    float p_in = (float) nlo / (float) nsamp;

    if (ja98)
    {
      ja98->set_Phi (p_in);
      
      table->set_lo_val ( ja98->get_lo() );
      table->set_hi_val ( ja98->get_hi() );
      table->rebuild();
    }
    
    /* Generate the 256 sets of four output floating point values
       corresponding to each byte */
    table->generate ( lookup );
    lookup += lookup_block_size;
  }
}

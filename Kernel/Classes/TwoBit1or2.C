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
  nlow = nlow_min = nlow_max = 0;
  lookup_base = 0;
  temp_values = 0;
  ndim_per_digitizer = 1;
}

dsp::TwoBit1or2::~TwoBit1or2 ()
{
  destroy ();
}

void dsp::TwoBit1or2::destroy ()
{
  if (lookup_base) delete [] lookup_base; lookup_base = 0;
  if (temp_values) delete [] temp_values; temp_values = 0;
}

void dsp::TwoBit1or2::set_nlow_min (unsigned min)
{
  nlow_min = min;
  destroy ();
}

void dsp::TwoBit1or2::set_nlow_max (unsigned max)
{
  nlow_max = max;
  destroy ();
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

void dsp::TwoBit1or2::lookup_build (unsigned nsamp, unsigned ndim,
				    TwoBitTable* table,
				    JenetAnderson98* ja98)
{
  destroy ();

  assert (nlow_max > nlow_min);
  assert (nsamp > nlow_max);

  lookup_base = new float [(nlow_max - nlow_min + 1) * 4];
  temp_values = new unsigned char [nsamp * ndim];

  ndim_per_digitizer = ndim;

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
    }
    
    // Generate the four output levels corresponding to each 2-bit number
    table->generate_unique_values ( lookup );
    lookup += table->get_unique_values();
  }
}


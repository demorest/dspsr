/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/TwoBitLookup.h"
#include "JenetAnderson98.h"

#include <assert.h>

dsp::TwoBitLookup::TwoBitLookup ()
{
  ndat = nlow = nlow_min = nlow_max = 0;
  lookup_base = 0;
  ndim = 1;
}

dsp::TwoBitLookup::~TwoBitLookup ()
{
  destroy ();
}

void dsp::TwoBitLookup::destroy ()
{
  if (lookup_base)
    delete [] lookup_base;

  lookup_base = 0;
}

void dsp::TwoBitLookup::set_nlow_min (unsigned min)
{
  nlow_min = min;
  destroy ();
}

void dsp::TwoBitLookup::set_nlow_max (unsigned max)
{
  nlow_max = max;
  destroy ();
}

void dsp::TwoBitLookup::set_ndat (unsigned _ndat)
{
  ndat = _ndat;
  destroy ();
}

//! Set the dimension of the time samples (1=real, 2=complex)
void dsp::TwoBitLookup::set_ndim (unsigned _ndim)
{
  ndim = _ndim;
  destroy ();
}

void dsp::TwoBitLookup::create ()
{
  lookup_base = new float [(nlow_max - nlow_min + 1)*get_lookup_block_size()];
}

void dsp::TwoBitLookup::lookup_build (TwoBitTable* table,
				      JenetAnderson98* ja98)
{
  destroy ();
  create ();

  assert (nlow_max > nlow_min);
  assert (ndat >= nlow_max);

  float* lookup = lookup_base;

  for (unsigned nlo = nlow_min; nlo <= nlow_max; nlo++)
  {
    /* Refering to JA98, nlo is the number of samples between x2 and x4, 
       and p_in is the left-hand side of Eq.44 */

    unsigned use_nlow = nlo;
    if (nlo == 0)
      use_nlow = 1;
    if (nlow == ndat)
      use_nlow = ndat - 1;

    float p_in = (float) use_nlow / (float) ndat;

    if (ja98)
    {
      ja98->set_Phi (p_in);
      
      table->set_lo_val ( ja98->get_lo() );
      table->set_hi_val ( ja98->get_hi() );
    }
    
    get_lookup_block (lookup, table);
    lookup += get_lookup_block_size ();
  }
}

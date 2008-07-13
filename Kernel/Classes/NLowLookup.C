/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/NLowLookup.h"

dsp::NLowLookup::NLowLookup (BitTable* table)
{
  bit_table = table;

  lookup = table->get_values ();
  table->get_nlow_lookup (nlow_lookup);
}

//! Return the pointer to the bit table
const dsp::BitTable* dsp::NLowLookup::get_table () const
{
  return bit_table;
}


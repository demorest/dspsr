/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/APSREightBit.h"
#include "apsr_unpack.h"

#include "dsp/Observation.h"
#include "dsp/BitTable.h"

bool dsp::APSREightBit::matches (const Observation* observation)
{
  return observation->get_machine() == "APSR"
    && observation->get_nbit() == 8
    && observation->get_state() == Signal::Analytic;
}

//! Null constructor
dsp::APSREightBit::APSREightBit ()
  : EightBitUnpacker ("APSREightBit")
{
  bool reverse_bits = true;
  table = new BitTable (8, BitTable::TwosComplement, reverse_bits);
}

/*!
  The real and imaginary components of the complex polyphase
  filterbank outputs are decimated together
*/
unsigned dsp::APSREightBit::get_ndim_per_digitizer () const
{
  return 2;
}

void dsp::APSREightBit::unpack ()
{
  apsr_unpack (input, output, this);
}


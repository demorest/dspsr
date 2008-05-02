/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/APSRFourBit.h"
#include "dsp/Observation.h"
#include "dsp/BitTable.h"

#include "apsr_unpack.h"

bool dsp::APSRFourBit::matches (const Observation* observation)
{
  return observation->get_machine() == "APSR"
    && observation->get_nbit() == 4
    && observation->get_state() == Signal::Analytic;
}

//! Null constructor
dsp::APSRFourBit::APSRFourBit ()
  : FourBitUnpacker ("APSRFourBit")
{
  bool reverse_bits = false;
  table = new BitTable (4, BitTable::TwosComplement, reverse_bits);
}

/*!
  The real and imaginary components of the complex polyphase
  filterbank outputs are decimated together
*/
unsigned dsp::APSRFourBit::get_ndim_per_digitizer () const
{
  return 2;
}

void dsp::APSRFourBit::unpack ()
{
  apsr_unpack (input, output, this);
}


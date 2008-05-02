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

using namespace std;

bool dsp::APSREightBit::matches (const Observation* observation)
{
  return observation->get_machine() == "APSR"
    && observation->get_nbit() == 8;
}

//! Null constructor
dsp::APSREightBit::APSREightBit ()
  : EightBitUnpacker ("APSREightBit")
{
  bool reverse_bits = false;
  table = new BitTable (8, BitTable::TwosComplement, reverse_bits);
}

/*!
  The real and imaginary components of the complex polyphase
  filterbank outputs are decimated together
*/
unsigned dsp::APSREightBit::get_ndim_per_digitizer () const
{
  return input->get_ndim();
}

void dsp::APSREightBit::unpack ()
{
  if (input->get_npol() == 1)     // Jayanta's test data has one poln
    BitUnpacker::unpack ();
  else
    apsr_unpack (input, output, this);
}


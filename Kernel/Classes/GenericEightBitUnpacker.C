/***************************************************************************
 *
 *   Copyright (C) 2012 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/GenericEightBitUnpacker.h"
#include "dsp/BitTable.h"

//! Constructor
dsp::GenericEightBitUnpacker::GenericEightBitUnpacker ()
  : EightBitUnpacker ("GenericEightBitUnpacker")
{
  table = new BitTable (8, BitTable::TwosComplement);
}

bool dsp::GenericEightBitUnpacker::matches (const Observation* observation)
{
  return observation->get_nbit() == 8;
}


/***************************************************************************
 *
 *   Copyright (C) 2005 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/PuMa2Unpacker.h"
#include "dsp/BitTable.h"

//! Constructor
dsp::PuMa2Unpacker::PuMa2Unpacker (const char* name) : EightBitUnpacker (name)
{
  table = new BitTable (8, BitTable::TwosComplement);
}

bool dsp::PuMa2Unpacker::matches (const Observation* observation)
{
  return observation->get_machine() == "PuMa2" 
    && observation->get_nbit() == 8;
//    && observation->get_state() == Signal::Nyquist;
}


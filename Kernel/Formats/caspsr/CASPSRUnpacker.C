/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/CASPSRUnpacker.h"
#include "dsp/BitTable.h"

//! Constructor
dsp::CASPSRUnpacker::CASPSRUnpacker (const char* name) : EightBitUnpacker (name)
{
  table = new BitTable (8, BitTable::TwosComplement);
}

bool dsp::CASPSRUnpacker::matches (const Observation* observation)
{
  return observation->get_machine() == "CASPSR" 
    && observation->get_nbit() == 8
    && observation->get_state() == Signal::Analytic;
}


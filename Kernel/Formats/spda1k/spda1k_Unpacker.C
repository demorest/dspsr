/***************************************************************************
 *
 *   Copyright (C) 2009 by Aidan Hotan
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/spda1k_Unpacker.h"
#include "dsp/EightBitUnpacker.h"

bool dsp::SPDA1K_Unpacker::matches (const Observation* observation)
{
  return observation->get_machine() == "SPDA1K"
    && observation->get_nbit() == 8
    && observation->get_state() == Signal::Nyquist;
}

//! Null constructor
dsp::SPDA1K_Unpacker::SPDA1K_Unpacker ()
  : EightBitUnpacker ("SPDA1K_Unpacker")
{
  table = new BitTable (8, BitTable::TwosComplement);
}


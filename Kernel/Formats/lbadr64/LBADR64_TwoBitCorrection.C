/***************************************************************************
 *
 *   Copyright (C) 2004 by Craig West & Aidan Hotan
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/LBADR64_TwoBitCorrection.h"
#include "dsp/TwoBitTable.h"

bool dsp::LBADR64_TwoBitCorrection::matches (const Observation* observation)
{
  return observation->get_machine() == "LBADR64"
    && observation->get_nbit() == 2
    && observation->get_state() == Signal::Nyquist;
}

//! Null constructor
dsp::LBADR64_TwoBitCorrection::LBADR64_TwoBitCorrection ()
  : TwoBitCorrection ("LBADR64_TwoBitCorrection")
{
  table = new TwoBitTable (TwoBitTable::SignMagnitude);
  table->set_reverse_2bit ();
}


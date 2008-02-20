/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/APSREightBit.h"
#include "dsp/Observation.h"
#include "dsp/EightBitTable.h"

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
  table = new EightBitTable (EightBitTable::TwosComplement);
}












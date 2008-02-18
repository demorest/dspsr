/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/APSRFourBit.h"
#include "dsp/Observation.h"
#include "dsp/FourBitTable.h"

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
  table = new FourBitTable (FourBitTable::TwosComplement);
}












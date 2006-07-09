/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/CPSR2TwoBitCorrection.h"
#include "dsp/Observation.h"
#include "dsp/TwoBitTable.h"

bool dsp::CPSR2TwoBitCorrection::matches (const Observation* observation)
{
  return observation->get_machine() == "CPSR2"
    && observation->get_nbit() == 2
    && observation->get_state() == Signal::Nyquist;
}

//! Null constructor
dsp::CPSR2TwoBitCorrection::CPSR2TwoBitCorrection ()
  : TwoBitCorrection ("CPSR2TwoBitCorrection")
{
  table = new TwoBitTable (TwoBitTable::OffsetBinary);
}












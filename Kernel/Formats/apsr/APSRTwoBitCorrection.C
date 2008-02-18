/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/APSRTwoBitCorrection.h"
#include "dsp/Observation.h"
#include "dsp/TwoBitTable.h"

bool dsp::APSRTwoBitCorrection::matches (const Observation* observation)
{
  return observation->get_machine() == "APSR"
    && observation->get_nbit() == 2
    && observation->get_state() == Signal::Nyquist;
}

//! Null constructor
dsp::APSRTwoBitCorrection::APSRTwoBitCorrection ()
  : TwoBitCorrection ("APSRTwoBitCorrection")
{
  table = new TwoBitTable (TwoBitTable::OffsetBinary);
}












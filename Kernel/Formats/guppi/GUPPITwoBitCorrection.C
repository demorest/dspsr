/***************************************************************************
 *
 *   Copyright (C) 2013 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/GUPPITwoBitCorrection.h"
#include "dsp/Observation.h"
#include "dsp/TwoBitTable.h"

bool dsp::GUPPITwoBitCorrection::matches (const Observation* observation)
{
  return observation->get_machine().substr(1) == "UPPI"
    && observation->get_nbit() == 2
    && observation->get_state() == Signal::Nyquist;
}

//! Null constructor
dsp::GUPPITwoBitCorrection::GUPPITwoBitCorrection ()
  : TwoBitCorrection ("GUPPITwoBitCorrection")
{
  table = new TwoBitTable (TwoBitTable::OffsetBinary);
  // This tells unpacker that LSB is first sample in time order
  table->set_order(TwoBitTable::LeastToMost); 
}


/***************************************************************************
 *
 *   Copyright (C) 2011 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/VDIFTwoBitCorrection.h"
#include "dsp/TwoBitTable.h"

bool dsp::VDIFTwoBitCorrection::matches (const Observation* observation)
{
  return observation->get_machine() == "VDIF" && observation->get_nbit() == 2;
}

//! Null constructor
dsp::VDIFTwoBitCorrection::VDIFTwoBitCorrection ()
  : TwoBitCorrection ("VDIFTwoBitCorrection")
{
  table = new TwoBitTable (TwoBitTable::OffsetBinary);
  set_ndig(1);
}


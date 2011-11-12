/***************************************************************************
 *
 *   Copyright (C) 2011 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/VDIFTwoBitCorrection.h"
#include "dsp/TwoBitTable.h"
#include "dsp/VDIFTwoBitTable.h"

bool dsp::VDIFTwoBitCorrection::matches (const Observation* observation)
{
  return observation->get_machine() == "VDIF" 
      && observation->get_nbit() == 2
      && observation->get_npol() == 1;
}

//! Null constructor
dsp::VDIFTwoBitCorrection::VDIFTwoBitCorrection ()
  : TwoBitCorrection ("VDIFTwoBitCorrection")
{
  table = new VDIFTwoBitTable (TwoBitTable::OffsetBinary);
  set_ndig(1);
}


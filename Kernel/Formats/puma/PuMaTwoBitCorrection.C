/***************************************************************************
 *
 *   Copyright (C) 2003 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/PuMaTwoBitCorrection.h"
#include "dsp/TwoBitTable.h"

bool dsp::PuMaTwoBitCorrection::matches (const Observation* observation)
{
  return observation->get_machine() == "PuMa" && observation->get_nbit() == 2;
}

//! Null constructor
dsp::PuMaTwoBitCorrection::PuMaTwoBitCorrection ()
  : SubByteTwoBitCorrection ("PuMaTwoBitCorrection")
{
  threshold = 1.5;
  table = new TwoBitTable (TwoBitTable::OffsetBinary);
}


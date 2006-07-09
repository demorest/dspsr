/***************************************************************************
 *
 *   Copyright (C) 2004 by Craig West
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include <stdio.h>

#include "dsp/Mark4TwoBitCorrection.h"
#include "dsp/Mark4TwoBitTable.h"
#include "dsp/Observation.h"

bool dsp::Mark4TwoBitCorrection::matches (const Observation* observation)
{
  return observation->get_machine() == "Mark4"
    && observation->get_nbit() == 2
    && observation->get_state() == Signal::Nyquist;
}

//! Null constructor
dsp::Mark4TwoBitCorrection::Mark4TwoBitCorrection ()
  : TwoBitCorrection ("Mark4TwoBitCorrection")
{
  table = new Mark4TwoBitTable (TwoBitTable::OffsetBinary);
  //  table->set_flip(true);
}

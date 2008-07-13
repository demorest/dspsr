/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/APSRTwoBitCorrection.h"

dsp::APSRTwoBitCorrection::APSRTwoBitCorrection ()
  : APSRUnpacker<TwoBitCorrection,2> ("APSRTwoBitCorrection")
{
  table = new TwoBitTable (TwoBitTable::TwosComplement);
}


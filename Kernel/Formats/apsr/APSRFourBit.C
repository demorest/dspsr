/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/APSRFourBit.h"

dsp::APSRFourBit::APSRFourBit ()
  : APSRFourBitBase ("APSRFourBit", new BitTable (4, BitTable::TwosComplement))
{
}


/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/APSREightBit.h"

dsp::APSREightBit::APSREightBit ()
  : APSREightBitBase ("APSREightBit", new BitTable (8, BitTable::TwosComplement))
{
}


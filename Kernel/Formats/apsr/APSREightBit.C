/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/APSREightBit.h"

dsp::BitTable* eight_bit_six ()
{
  dsp::BitTable* table = new dsp::BitTable (8, dsp::BitTable::TwosComplement);
  table->set_effective_nbit (6);
  return table;
}

dsp::APSREightBit::APSREightBit ()
  : APSREightBitBase ("APSREightBit", eight_bit_six())
{
}


/***************************************************************************
 *
 *   Copyright (C) 2011 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/VDIFTwoBitTable.h"

unsigned dsp::VDIFTwoBitTable::extract (unsigned byte, unsigned sample) const
{
  unsigned char shifts[4] = { 0, 2, 4, 6 };
  return byte >> shifts[sample] & 0x03;
}

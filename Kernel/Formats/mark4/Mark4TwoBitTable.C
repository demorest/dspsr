/***************************************************************************
 *
 *   Copyright (C) 2004 by Craig West
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/Mark4TwoBitTable.h"

/*! Each byte is interpreted as follows:

  MSB: l3 l2 l1 l0 m3 m2 m1 m0 :LSB

  Where each two bit timesample is given by its most significant bit, mX,
  and least significant bit, lX, and each byte contains four time samples:
  m0l0, m1l1, m2l2, m3l3.

  \param byte the byte pattern containing four time samples
  \param sample the sample to extract from the byte 
*/
unsigned dsp::Mark4TwoBitTable::twobit (unsigned byte, unsigned sample) const
{
  unsigned char shifts[4] = { 0, 1, 2, 3 };

  unsigned lsb = (byte >> (shifts[sample]+4)) & 0x01;
  unsigned msb = (byte >> shifts[sample]) & 0x01;

  return lsb | (msb << 1);
}

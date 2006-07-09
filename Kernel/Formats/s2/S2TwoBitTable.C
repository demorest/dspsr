/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/S2TwoBitTable.h"

/*! Each byte is interpreted as follows:

  MSB: l3 l1 l2 l0 m3 m1 m2 m0 :LSB

  Where each two bit timesample is given by its most significant bit, mX,
  and least significant bit, lX, and each byte contains four time samples:
  m0l0, m1l1, m2l2, m3l3.

  \param byte the byte pattern containing four time samples
  \param sample the sample to extract from the byte 
*/
unsigned dsp::S2TwoBitTable::twobit (unsigned byte, unsigned sample) const
{
  unsigned char shifts[4] = { 0, 2, 1, 3 };

  unsigned lsb = (byte >> (shifts[sample]+4)) & 0x01;
  unsigned msb = (byte >> shifts[sample]) & 0x01;

  return lsb | (msb << 1);
}

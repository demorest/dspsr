#include "S2TwoBitTable.h"

/*! Each byte is interpreted as follows:

  MSB: m4 m2 m3 m1 l4 l2 l3 l1 :LSB

  Where each two bit timesample is given by its most significant bit, mX,
  and least significant bit, lX, and each byte contains four time samples:
  m1l1, m2l2, m3l3, m4l4.

  \param byte the byte pattern containing four time samples
  \param sample the sample to extract from the byte 
*/
unsigned dsp::TwoBitTable::twobit (unsigned byte, unsigned sample) const
{
  unsigned char shifts[4] = { 0, 2, 1, 3 };

  unsigned char lsb = 0x01;
  unsigned char msb = 0x02;

  return ((byte>>shifts[sample]) & lsb) | ((byte>>(shifts[sample]+3)) & msb);
}

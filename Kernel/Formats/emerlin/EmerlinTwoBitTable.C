/***************************************************************************
 *
 *   Copyright (C) 2011 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/EmerlinTwoBitTable.h"
#include <iostream>

unsigned dsp::EmerlinTwoBitTable::extract (unsigned byte, unsigned sample) const
{
  unsigned char shifts[4] = { 0, 2, 4, 6 }; // LSB is first sample. VDIF standard
//  unsigned char shifts[4] = { 6, 4, 2, 0 };
  //std::cout << "dsp::EmerlinTwoBitTable::extract()" << std::endl;
  return byte >> shifts[sample] & 0x03;
}

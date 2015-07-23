//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2015 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __GUPPIFourBit_h
#define __GUPPIFourBit_h

#include "dsp/FourBitUnpacker.h"

namespace dsp
{
  //! Converts GUPPI data from 4-bit to floating point values
  // This class is mainly meant for 4-bit VDIF data from the phased
  // VLA that has been repacked into GUPPI block format.  No other
  // GUPPI-based instruments produce 4-bit data as far as I know.
  class GUPPIFourBit: public FourBitUnpacker
  {
  public:

    //! Constructor initializes bit table
    GUPPIFourBit ();

    //! Return true if this unpacker can handle the observation
    bool matches (const Observation*);

  };
}

#endif

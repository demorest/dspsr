//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __APSRFourBit_h
#define __APSRFourBit_h

#include "dsp/APSRUnpacker.h"
#include "dsp/FourBitTwo.h"

namespace dsp
{
  typedef APSRUnpacker< APSRExcision<FourBitTwo>, 4 > APSRFourBitBase;

  //! Converts APSR data from 8-bit digitized to floating point values
  class APSRFourBit: public APSRFourBitBase
  {
  public:
    //! Constructor initializes bit table
    APSRFourBit ();
  };
}

#endif

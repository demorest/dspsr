//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __APSREightBit_h
#define __APSREightBit_h

#include "dsp/APSRUnpacker.h"
#include "dsp/EightBitOne.h"

namespace dsp
{
  typedef APSRUnpacker< APSRExcision<EightBitOne>, 8 > APSREightBitBase;

  //! Converts APSR data from 8-bit digitized to floating point values
  class APSREightBit: public APSREightBitBase
  {
  public:
    //! Constructor initializes bit table
    APSREightBit ();
  };
}

#endif

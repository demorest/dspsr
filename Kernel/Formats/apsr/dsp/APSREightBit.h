//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __APSREightBit_h
#define __APSREightBit_h

#include "dsp/EightBitUnpacker.h"

namespace dsp {

  //! Converts APSR data from 4-bit digitized to floating point values
  class APSREightBit: public EightBitUnpacker {

  public:

    //! Constructor initializes base class attributes
    APSREightBit ();

    //! Return true if APSREightBit can convert the Observation
    virtual bool matches (const Observation* observation);

  };
  
}

#endif

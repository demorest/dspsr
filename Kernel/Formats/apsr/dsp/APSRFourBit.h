//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __APSRFourBit_h
#define __APSRFourBit_h

#include "dsp/FourBitUnpacker.h"

namespace dsp {

  //! Converts APSR data from 4-bit digitized to floating point values
  class APSRFourBit: public FourBitUnpacker {

  public:

    //! Constructor initializes base class attributes
    APSRFourBit ();

    //! Return true if APSRFourBit can convert the Observation
    virtual bool matches (const Observation* observation);

  };
  
}

#endif

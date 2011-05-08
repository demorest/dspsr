//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __GMRTFourBit_h
#define __GMRTFourBit_h

#include "dsp/FourBitUnpacker.h"

namespace dsp
{
  //! Converts single-dish GMRT data from 4-bit to floating point values
  class GMRTFourBit: public FourBitUnpacker
  {
  public:

    //! Constructor initializes bit table
    GMRTFourBit ();

    //! Return true if this unpacker can handle the observation
    bool matches (const Observation*);

  };
}

#endif

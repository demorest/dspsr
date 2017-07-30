//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2017 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __GenericFourBitUnpacker_h
#define __GenericFourBitUnpacker_h

#include "dsp/FourBitUnpacker.h"

namespace dsp
{
  //! Converts single-dish GMRT data from 4-bit to floating point values
  class GenericFourBitUnpacker: public FourBitUnpacker
  {
  public:

    //! Constructor initializes bit table
    GenericFourBitUnpacker ();

    //! Return true if this unpacker can handle the observation
    bool matches (const Observation*);

  };
}

#endif

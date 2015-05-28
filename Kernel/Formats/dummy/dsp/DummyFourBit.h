//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2015 by Erik Madsen (from GUPPIFourBit by P. Demorest)
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __DummyFourBit_h
#define __DummyFourBit_h

#include "dsp/FourBitUnpacker.h"

namespace dsp
{
  // Based on GUPPIFourBit
  class DummyFourBit: public FourBitUnpacker
  {
  public:

    //! Constructor initializes bit table
    DummyFourBit ();

    //! Return true if this unpacker can handle the observation
    bool matches (const Observation*);

  };
}

#endif

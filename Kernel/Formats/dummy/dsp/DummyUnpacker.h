//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2015 by Erik Madsen (based on GMRTUnpacker)
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __DummyUnpacker_h
#define __DummyUnpacker_h

#include "dsp/EightBitUnpacker.h"

namespace dsp {

  //! Simple 8-bit to float unpacker
  class DummyUnpacker : public EightBitUnpacker {

  public:
    
    //! Constructor
    DummyUnpacker (const char* name = "DummyUnpacker");

   protected:
    
    //! Return true if we can convert the Observation
    virtual bool matches (const Observation* observation);

  };

}

#endif // !defined(__DummyUnpacker_h)

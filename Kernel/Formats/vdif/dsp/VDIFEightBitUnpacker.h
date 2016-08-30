//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2015 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __VDIFEightBitUnpacker_h
#define __VDIFEightBitUnpacker_h

#include "dsp/EightBitUnpacker.h"

namespace dsp {

  //! Unpack 8-bit VDIF data
  class VDIFEightBitUnpacker : public EightBitUnpacker {

  public:
    
    //! Constructor
    VDIFEightBitUnpacker (const char* name = "VDIFEightBitUnpacker");

   protected:
    
    //! Return true if we can convert the Observation
    virtual bool matches (const Observation* observation);

  };

}

#endif // !defined(__VDIFEightBitUnpacker_h)

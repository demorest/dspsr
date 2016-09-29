//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2016 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __VDIFFourBitUnpacker_h
#define __VDIFFourBitUnpacker_h

#include "dsp/FourBitUnpacker.h"

namespace dsp {

  //! Unpack 4-bit, single-pol VDIF data
  class VDIFFourBitUnpacker : public FourBitUnpacker {

  public:
    
    //! Constructor
    VDIFFourBitUnpacker (const char* name = "VDIFFourBitUnpacker");

   protected:
    
    //! Return true if we can convert the Observation
    virtual bool matches (const Observation* observation);

  };

}

#endif // !defined(__VDIFEightBitUnpacker_h)

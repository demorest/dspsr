//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2006 by Jayanta Roy and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Formats/gmrt/dsp/GMRTUnpacker.h

#ifndef __GMRTUnpacker_h
#define __GMRTUnpacker_h

#include "dsp/EightBitUnpacker.h"

namespace dsp {

  //! Simple 8-bit to float unpacker for the GMRT files
  class GMRTUnpacker : public EightBitUnpacker {

  public:
    
    //! Constructor
    GMRTUnpacker (const char* name = "GMRTUnpacker");

   protected:
    
    //! Return true if we can convert the Observation
    virtual bool matches (const Observation* observation);

  };

}

#endif // !defined(__GMRTUnpacker_h)

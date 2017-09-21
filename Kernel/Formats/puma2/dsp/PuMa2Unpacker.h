//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2005 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Formats/puma2/dsp/PuMa2Unpacker.h

#ifndef __PuMa2Unpacker_h
#define __PuMa2Unpacker_h

#include "dsp/EightBitUnpacker.h"

namespace dsp {

  //! Simple 8-bit to float unpacker for the PuMa2 files
  class PuMa2Unpacker : public EightBitUnpacker
  {

  public:
    
    //! Constructor
    PuMa2Unpacker (const char* name = "PuMa2Unpacker");

   protected:

    //! Return true if we can convert the Observation
    virtual bool matches (const Observation* observation);

  };

}

#endif // !defined(__PuMa2Unpacker_h)


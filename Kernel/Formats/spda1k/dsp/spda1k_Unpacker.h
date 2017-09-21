//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2009 by Aidan Hotan
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Formats/spda1k/dsp/spda1k_Unpacker.h

#ifndef __SPDA1K_Unpacker_h
#define __SPDA1K_Unpacker_h

#include "dsp/EightBitUnpacker.h"
#include "dsp/BitTable.h"

namespace dsp {
  
  class SPDA1K_Unpacker: public EightBitUnpacker {

  public:

    //! Constructor initializes base class atributes
    SPDA1K_Unpacker ();

    //! Return true if SPDA1K_Unpacker can convert the Observation
    virtual bool matches (const Observation* observation);

  };
  
}

#endif

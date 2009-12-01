//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2009 by Aidan Hotan
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/spda1k/dsp/spda1k_Unpacker.h,v $
   $Revision: 1.1 $
   $Date: 2009/12/01 07:55:12 $
   $Author: ahotan $ */

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

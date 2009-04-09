//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/caspsr/dsp/CASPSRUnpacker.h,v $
   $Revision: 1.1 $
   $Date: 2009/04/09 02:06:40 $
   $Author: straten $ */

#ifndef __CASPSRUnpacker_h
#define __CASPSRUnpacker_h

#include "dsp/EightBitUnpacker.h"

namespace dsp {

  //! Simple 8-bit to float unpacker for CASPSR files
  class CASPSRUnpacker : public EightBitUnpacker
  {

  public:
    
    //! Constructor
    CASPSRUnpacker (const char* name = "CASPSRUnpacker");

   protected:

    //! Return true if we can convert the Observation
    virtual bool matches (const Observation* observation);

  };

}

#endif // !defined(__CASPSRUnpacker_h)


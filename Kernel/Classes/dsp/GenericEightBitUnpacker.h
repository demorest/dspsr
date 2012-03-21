//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2012 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/GenericEightBitUnpacker.h,v $
   $Revision: 1.1 $
   $Date: 2012/03/21 09:19:09 $
   $Author: straten $ */

#ifndef __GenericEightBitUnpacker_h
#define __GenericEightBitUnpacker_h

#include "dsp/EightBitUnpacker.h"

namespace dsp {

  //! Simple 8-bit to float unpacker for generic 8-bit files
  class GenericEightBitUnpacker : public EightBitUnpacker
  {

  public:
    
    //! Constructor
    GenericEightBitUnpacker ();

   protected:

    //! Return true if this unpacker can convert the Observation
    virtual bool matches (const Observation* observation);

  };

}

#endif // !defined(__GenericEightBitUnpacker_h)


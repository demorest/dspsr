//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2004 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/EightBitUnpacker.h,v $
   $Revision: 1.4 $
   $Date: 2008/04/15 08:11:41 $
   $Author: straten $ */

#ifndef __EightBitUnpacker_h
#define __EightBitUnpacker_h

#include "dsp/BitUnpacker.h"

namespace dsp {

  //! Converts 4-bit digitised samples to floating point
  class EightBitUnpacker: public BitUnpacker
  {

  public:

    //! Null constructor
    EightBitUnpacker (const char* name = "EightBitUnpacker");

  protected:

    void unpack (uint64 ndat, const unsigned char* from, const unsigned nskip,
		 float* into, const unsigned fskip, unsigned long* hist);

  };
}

#endif

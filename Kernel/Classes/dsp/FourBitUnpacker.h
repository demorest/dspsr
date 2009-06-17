//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2004 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/FourBitUnpacker.h,v $
   $Revision: 1.9 $
   $Date: 2009/06/17 10:16:53 $
   $Author: straten $ */

#ifndef __FourBitUnpacker_h
#define __FourBitUnpacker_h

#include "dsp/BitUnpacker.h"

namespace dsp {

  //! Converts 4-bit digitised samples to floating point
  class FourBitUnpacker: public BitUnpacker
  {

  public:

    //! Null constructor
    FourBitUnpacker (const char* name = "FourBitUnpacker");

    //! Get the histogram for the specified digitizer
    void get_histogram (std::vector<unsigned long>&, unsigned idig) const;

  protected:

    void unpack (uint64_t ndat, const unsigned char* from, const unsigned nskip,
		 float* into, const unsigned fskip, unsigned long* hist);

  };
}
#endif

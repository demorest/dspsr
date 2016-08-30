//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2014 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __GUPPIFITS_unpack_h
#define __GUPPIFITS_unpack_h

#include "dsp/Unpacker.h"
#include "fitsio.h"

#include "dsp/TimeSeries.h"
#include "dsp/BitSeries.h"

namespace dsp
{
  //! Unpacks data from PSRFITS files produced by GUPPI and related systems
  /*! This class unpacks search-mode PSRFITS data from GUPPI and similar
   * systems.  These systems used a different sign and zero offset
   * convention from what is implmented in the standard FITSUnpacker
   * class.  It may be possible to merge this code back in so that 
   * a single class handles both, but for now this seemed simpler.
   */
  class GUPPIFITSUnpacker : public Unpacker
  {
    public:
      GUPPIFITSUnpacker(const char* name = "GUPPIFITSUnpacker");

    protected:
      virtual void unpack();

      virtual bool matches(const Observation* observation);
  };
}

#endif

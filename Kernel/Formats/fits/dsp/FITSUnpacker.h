//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Jonathan Khoo
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __FITS_unpack_h
#define __FITS_unpack_h

#include "dsp/Unpacker.h"
#include "fitsio.h"

#include "dsp/TimeSeries.h"
#include "dsp/BitSeries.h"

namespace dsp
{
  class FITSUnpacker : public Unpacker
  {
    public:
      FITSUnpacker(const char* name = "FITSUnpacker");

    protected:
      virtual void unpack();

      virtual bool matches(const Observation* observation);

      float oneBitNumber(const int num);

      float twoBitNumber(const int num);

      float fourBitNumber(const int num);

      float eightBitNumber(const int num);
  };
}

#endif

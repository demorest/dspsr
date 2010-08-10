//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Jonathan Khoo
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __FITS_unpack_h
#define __FITS_unpack_h

#include "dsp/HistUnpacker.h"
#include "fitsio.h"

namespace dsp
{
    class FITSUnpacker : public HistUnpacker
    {
        public:
            FITSUnpacker(const char* name = "FITSUnpacker");

        protected:
            virtual void unpack();

            virtual bool matches(const Observation* observation);
    };
}

float oneBitNumber(int num);
float eightBitNumber(int num);
float fourBitNumber(int num);
float twoBitNumber(int num);

#endif

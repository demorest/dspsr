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

float getChan(fitsfile *fp, const int sample, const int subint, const int npol, const int pol,
        const int samplesperbyte, const int nchan, const int chan, const int colnum);

float oneBitNumber(int num);
float eightBitNumber(int num);
float fourBitNumber(int num);
float twoBitNumber(int num);

#endif

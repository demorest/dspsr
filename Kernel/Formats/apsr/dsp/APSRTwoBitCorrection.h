//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Formats/apsr/dsp/APSRTwoBitCorrection.h

#ifndef __APSRTwoBitCorrection_h
#define __APSRTwoBitCorrection_h

#include "dsp/APSRUnpacker.h"
#include "dsp/TwoBitCorrection.h"

namespace dsp
{
  //! Converts APSR data from 2-bit digitized to floating point values
  class APSRTwoBitCorrection: public APSRUnpacker<TwoBitCorrection,2>
  {
  public:
    //! Constructor initializes two bit table
    APSRTwoBitCorrection ();
  };
}

#endif


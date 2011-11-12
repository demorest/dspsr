//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __VDIFTwoBitCorrectionMulti_h
#define __VDIFTwoBitCorrectionMulti_h

#include "dsp/SubByteTwoBitCorrection.h"

namespace dsp {
  
  //! Converts VDIF data from 2-bit digitized to floating point values
  class VDIFTwoBitCorrectionMulti: public SubByteTwoBitCorrection {

  public:

    //! Constructor initializes base class atributes
    VDIFTwoBitCorrectionMulti ();

    //! Return true if VDIFTwoBitCorrection can convert the Observation
    virtual bool matches (const Observation* observation);

  };
  
}

#endif

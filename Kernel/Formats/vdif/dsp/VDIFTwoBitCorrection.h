//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __VDIFTwoBitCorrection_h
#define __VDIFTwoBitCorrection_h

#include "dsp/TwoBitCorrection.h"

namespace dsp {
  
  //! Converts VDIF data from 2-bit digitized to floating point values
  class VDIFTwoBitCorrection: public TwoBitCorrection {

  public:

    //! Constructor initializes base class atributes
    VDIFTwoBitCorrection ();

    //! Return true if VDIFTwoBitCorrection can convert the Observation
    virtual bool matches (const Observation* observation);

  };
  
}

#endif

//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2013 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __GUPPITwoBitCorrection_h
#define __GUPPITwoBitCorrection_h

class GUPPITwoBitCorrection;

#include "dsp/TwoBitCorrection.h"

namespace dsp {

  class TwoBitTable;

  //! Converts GUPPI data from 2-bit digitized to floating point values
  class GUPPITwoBitCorrection: public TwoBitCorrection {

  public:

    //! Constructor initializes base class attributes
    GUPPITwoBitCorrection ();

    //! Return true if GUPPITwoBitCorrection can convert the Observation
    virtual bool matches (const Observation* observation);

  };
  
}

#endif

//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2004 by Craig West
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Formats/mark4/dsp/Mark4TwoBitCorrection.h

#ifndef __Mark4TwoBitCorrection_h
#define __Mark4TwoBitCorrection_h

class Mark4TwoBitCorrection;

#include "dsp/TwoBitCorrection.h"

namespace dsp {

  class TwoBitTable;

  //! Converts Mark4 data from 2-bit digitized to floating point values
  class Mark4TwoBitCorrection: public TwoBitCorrection {

  public:

    //! Constructor initializes base class attributes
    Mark4TwoBitCorrection ();

    //! Return true if Mark4TwoBitCorrection can convert the Observation
    virtual bool matches (const Observation* observation);

  };
  
}

#endif

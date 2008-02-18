//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/apsr/dsp/APSRTwoBitCorrection.h,v $
   $Revision: 1.1 $
   $Date: 2008/02/18 10:49:55 $
   $Author: straten $ */

#ifndef __APSRTwoBitCorrection_h
#define __APSRTwoBitCorrection_h

class APSRTwoBitCorrection;

#include "dsp/TwoBitCorrection.h"

namespace dsp {

  //! Converts APSR data from 2-bit digitized to floating point values
  class APSRTwoBitCorrection: public TwoBitCorrection {

  public:

    //! Constructor initializes base class attributes
    APSRTwoBitCorrection ();

    //! Return true if APSRTwoBitCorrection can convert the Observation
    virtual bool matches (const Observation* observation);

  };
  
}

#endif

//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/apsr/dsp/APSRTwoBitCorrection.h,v $
   $Revision: 1.4 $
   $Date: 2008/07/13 00:38:54 $
   $Author: straten $ */

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


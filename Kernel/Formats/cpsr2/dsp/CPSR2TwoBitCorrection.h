//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/cpsr2/dsp/CPSR2TwoBitCorrection.h,v $
   $Revision: 1.4 $
   $Date: 2002/10/07 01:48:37 $
   $Author: wvanstra $ */

#ifndef __CPSR2TwoBitCorrection_h
#define __CPSR2TwoBitCorrection_h

#include <vector>

#include "TwoBitCorrection.h"

namespace dsp {

  class TwoBitTable;

  //! Converts a CPSR2 Timeseries from 2-bit digitized to floating point values
  class CPSR2TwoBitCorrection: public TwoBitCorrection {

  public:

    //! Constructor
    CPSR2TwoBitCorrection ();

  };
  
}

#endif

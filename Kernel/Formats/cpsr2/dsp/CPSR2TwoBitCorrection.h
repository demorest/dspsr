//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/cpsr2/dsp/CPSR2TwoBitCorrection.h,v $
   $Revision: 1.6 $
   $Date: 2002/10/15 13:11:20 $
   $Author: pulsar $ */

#ifndef __CPSR2TwoBitCorrection_h
#define __CPSR2TwoBitCorrection_h

#include "TwoBitCorrection.h"

namespace dsp {

  class TwoBitTable;

  //! Converts CPSR2 data from 2-bit digitized to floating point values
  class CPSR2TwoBitCorrection: public TwoBitCorrection {

  public:

    //! Constructor initializes base class attributes
    CPSR2TwoBitCorrection ();

    //! Return true if CPSR2TwoBitCorrection can convert the Observation
    virtual bool matches (const Observation* observation);

  };
  
}

#endif

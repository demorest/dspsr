//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/cpsr2/dsp/CPSR2TwoBitCorrection.h,v $
   $Revision: 1.8 $
   $Date: 2002/11/06 06:30:41 $
   $Author: hknight $ */

#ifndef __CPSR2TwoBitCorrection_h
#define __CPSR2TwoBitCorrection_h

class CPSR2TwoBitCorrection;

#include "dsp/TwoBitCorrection.h"

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

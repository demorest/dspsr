//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/cpsr2/dsp/CPSR2TwoBitCorrection.h,v $
   $Revision: 1.9 $
   $Date: 2006/07/09 13:27:07 $
   $Author: wvanstra $ */

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

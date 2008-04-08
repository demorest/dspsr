//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/apsr/dsp/APSRTwoBitCorrection.h,v $
   $Revision: 1.3 $
   $Date: 2008/04/08 20:59:48 $
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

    unsigned get_ndim_per_digitizer () const;

    unsigned get_input_offset (unsigned idig) const;

    unsigned get_input_incr () const;

  };
  
}

#endif

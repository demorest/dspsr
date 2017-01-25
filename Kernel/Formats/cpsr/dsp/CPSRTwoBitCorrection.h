//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Formats/cpsr/dsp/CPSRTwoBitCorrection.h

#ifndef __CPSRTwoBitCorrection_h
#define __CPSRTwoBitCorrection_h

#include "dsp/SubByteTwoBitCorrection.h"

namespace dsp {
  
  //! Converts CPSR data from 2-bit digitized to floating point values
  class CPSRTwoBitCorrection: public SubByteTwoBitCorrection {

  public:

    //! Constructor initializes base class atributes
    CPSRTwoBitCorrection ();

    //! Return true if CPSRTwoBitCorrection can convert the Observation
    virtual bool matches (const Observation* observation);

    //! Over-ride the TwoBitCorrection default
    virtual unsigned get_ndig_per_byte () const;

    //! Over-ride the TwoBitCorrection default
    virtual unsigned get_output_offset (unsigned idig) const;

    //! Over-ride the TwoBitCorrection default
    virtual unsigned get_output_ipol (unsigned idig) const;

    //! Over-ride the TwoBitCorrection default
    virtual unsigned get_output_incr () const;

    //! Over-ride the SubByteTwoBitCorrection default
    virtual unsigned get_shift (unsigned idig, unsigned isamp) const;

  };
  
}

#endif

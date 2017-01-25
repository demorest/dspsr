/***************************************************************************
 *
 *   Copyright (C) 2006 by Eric Plum
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#ifndef __FadcTwoBitCorrection_h
#define __FadcTwoBitCorrection_h

#include "dsp/SubByteTwoBitCorrection.h"

namespace dsp {
  
  //! Converts FADC data from 2-bit digitized to floating point values
  class FadcTwoBitCorrection: public SubByteTwoBitCorrection {

  public:

    //! Constructor initializes base class atributes
    FadcTwoBitCorrection ();

    //! Return true if FadcTwoBitCorrection can convert the Observation
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

    //! Specialize the Unpacker for the Observation
    virtual void match (const Observation* observation);
  };
  
}

#endif

//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __VDIFTwoBitCorrectionMulti_h
#define __VDIFTwoBitCorrectionMulti_h

#include "dsp/SubByteTwoBitCorrection.h"

namespace dsp {

  //! Converts VDIF data from 2-bit digitized to floating point values
   /*! VDIF complex data is <most likely> to have a complex digitizer (ndim = 2)
   * but the default ndim_per_dig is 1. To support real and quadrature sampled (single pol) data
   * we need to override the 'get_ndim_from_digitizer' function here.
   * I am setting it to 2 at the moment. It is possible we may have to make this smarter
  */
 class VDIFTwoBitCorrectionMulti: public SubByteTwoBitCorrection {

  public:

    //! Constructor initializes base class atributes
    VDIFTwoBitCorrectionMulti ();

    //! Return true if VDIFTwoBitCorrection can convert the Observation
    virtual bool matches (const Observation* observation);
 
    //! Over-ride the TwoBitCorrection default
    virtual unsigned get_ndig_per_byte () const;

    //! Over-ride the TwoBitCorrection default
    virtual unsigned get_output_offset (unsigned idig) const;

    //! Over-ride the TwoBitCorrection default
    virtual unsigned get_output_ipol (unsigned idig) const;

    //! Over-ride the TwoBitCorrection default
    virtual unsigned get_output_incr () const;

    //! Over-ride the TwoBitCorrection default 
    virtual unsigned get_shift (unsigned idig, unsigned isamp) const;



  };
  
}

#endif

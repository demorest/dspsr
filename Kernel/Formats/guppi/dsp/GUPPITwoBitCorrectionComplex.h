//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2017 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __GUPPITwoBitCorrectionComplex_h
#define __GUPPITwoBitCorrectionComplex_h

#include "dsp/SubByteTwoBitCorrection.h"

namespace dsp {
  
  //! Converts GUPPI 2-pol, complex, 2-bit digitized to floating point values
  class GUPPITwoBitCorrectionComplex: public SubByteTwoBitCorrection {

  public:

    //! Constructor initializes base class atributes
    GUPPITwoBitCorrectionComplex ();

    //! Return true if GUPPITwoBitCorrection can convert the Observation
    virtual bool matches (const Observation* observation);

    //unsigned get_ndim_per_digitizer () const;
    unsigned get_ndig_per_byte () const;
    unsigned get_output_ichan (unsigned idig) const;
    unsigned get_output_ipol (unsigned idig) const;
    unsigned get_input_offset (unsigned idig) const;
    unsigned get_input_incr () const;
    unsigned get_shift (unsigned idig, unsigned isamp) const;

  };
  
}

#endif

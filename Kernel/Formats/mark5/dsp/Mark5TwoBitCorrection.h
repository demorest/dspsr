//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/mark5/dsp/Mark5TwoBitCorrection.h,v $
   $Revision: 1.2 $
   $Date: 2006/07/09 13:27:08 $
   $Author: wvanstra $ */

#ifndef __Mark5TwoBitCorrection_h
#define __Mark5TwoBitCorrection_h

#include "dsp/SubByteTwoBitCorrection.h"

namespace dsp {

  class Mark5File;

  //! Converts Mark5 data from 2-bit digitized to floating point values
  class Mark5TwoBitCorrection: public SubByteTwoBitCorrection {

  public:

    //! Constructor initializes base class atributes
    Mark5TwoBitCorrection ();

    //! Return true if Mark5TwoBitCorrection can convert the Observation
    virtual bool matches (const Observation* observation);

    //! Return true if Mark5TwoBitCorrection can convert the Observation
    static bool can_do (const Observation* observation);

    unsigned get_ndig_per_byte () const;

    unsigned get_input_incr () const;

    unsigned get_input_offset (unsigned idig) const;

    unsigned get_output_ipol (unsigned idig) const;

    unsigned get_output_ichan (unsigned idig) const;

  protected:

    //! Over-ride the SubByteTwoBitCorrection unpacking algorithm
    void dig_unpack (float* output_data,
		     const unsigned char* input_data, 
		     uint64 ndat,
		     unsigned digitizer,
		     unsigned* weights = 0,
		     unsigned nweights = 0);

    Mark5File* file;

  };
  
}

#endif

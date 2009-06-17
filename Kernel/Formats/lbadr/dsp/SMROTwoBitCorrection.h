//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2004 by Craig West
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/lbadr/dsp/SMROTwoBitCorrection.h,v $
   $Revision: 1.6 $
   $Date: 2009/06/17 10:16:54 $
   $Author: straten $ */

#ifndef __SMROTwoBitCorrection_h
#define __SMROTwoBitCorrection_h

#include "dsp/SubByteTwoBitCorrection.h"

namespace dsp {
  
  //! Converts SMRO data from 2-bit digitized to floating point values
  class SMROTwoBitCorrection: public SubByteTwoBitCorrection {

  public:

    //! Constructor initializes base class atributes
    SMROTwoBitCorrection ();

    //! Return true if SMROTwoBitCorrection can convert the Observation
    virtual bool matches (const Observation* observation);

    //! Over-ride the TwoBitCorrection default
    virtual unsigned get_ndig () const;
    
    //! Over-ride the TwoBitCorrection default
    virtual unsigned get_ndig_per_byte () const;

    //! Over-ride dig_unpack
    /*
    virtual void dig_unpack (float* output_data,
			     const unsigned char* input_data, 
			     uint64_t ndat,
			     unsigned digitizer,
			     unsigned* weights,
			     unsigned nweights);
    */
  };
  
}

#endif

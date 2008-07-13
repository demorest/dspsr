//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2003 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/SubByteTwoBitCorrection.h,v $
   $Revision: 1.6 $
   $Date: 2008/07/13 00:38:53 $
   $Author: straten $ */

#ifndef __SubByteTwoBitCorrection_h
#define __SubByteTwoBitCorrection_h

#include "dsp/TwoBitCorrection.h"
#include "dsp/TwoBitMask.h"
#include "dsp/TwoBit1or2.h"

namespace dsp {
  
  //! Converts BitSeries data from two-bit digitized to floating-point values
  /*! Use this class when the the digitized bits from different
    convertors are mixed within each byte. */

  class SubByteTwoBitCorrection: public TwoBitCorrection {

  public:

    //! Default constructor
    SubByteTwoBitCorrection (const char* name = "SubByteTwoBitCorrection");

    //! Destructor
    ~SubByteTwoBitCorrection ();

    //! Get the number of digitizer outputs in one byte
    virtual unsigned get_ndig_per_byte () const;

    //! Return the offset (number of bytes) into input for the given digitizer
    virtual unsigned get_input_offset (unsigned idig) const;

    //! Return the offset to the next byte of input data
    virtual unsigned get_input_incr () const;

    //! Return the bit shift for the given digitizer and the given sample
    virtual unsigned get_shift (unsigned idig, unsigned isamp) const;

  protected:

    //! Default unpacking algorithm
    void dig_unpack (const unsigned char* input_data, 
		     float* output_data,
		     uint64 ndat,
		     unsigned long* hist,
		     unsigned* weights = 0,
		     unsigned nweights = 0);

    TwoBit< 1, ShiftMask<1> > unpacker;

    //! Build the dynamic level setting lookup table and temporary space
    void build ();

  };
  
}

#endif

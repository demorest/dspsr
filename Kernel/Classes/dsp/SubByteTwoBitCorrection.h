//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/SubByteTwoBitCorrection.h,v $
   $Revision: 1.3 $
   $Date: 2006/02/23 17:51:58 $
   $Author: wvanstra $ */

#ifndef __SubByteTwoBitCorrection_h
#define __SubByteTwoBitCorrection_h

#include "dsp/TwoBitCorrection.h"

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

    //! Template unpacking algorithm
    template<class Rule>
    void dig_unpack (Rule& rule,
		     float* output_data,
		     const unsigned char* input_data, 
		     uint64 ndat,
		     unsigned digitizer,
		     unsigned* weights = 0,
		     unsigned nweights = 0);

    //! Default unpacking algorithm
    void dig_unpack (float* output_data,
		     const unsigned char* input_data, 
		     uint64 ndat,
		     unsigned digitizer,
		     unsigned* weights = 0,
		     unsigned nweights = 0);

    //! Temporary storage of bit-shifted values
    unsigned char* values;

    //! Two bit numbers equal to low-voltage state
    unsigned lovoltage [4];

    //! Build the dynamic level setting lookup table and temporary space
    void build ();

    //! Build the number of low-voltage states lookup table
    void nlo_build ();

    //! Delete allocated resources
    void destroy ();

  };
  
}

#endif

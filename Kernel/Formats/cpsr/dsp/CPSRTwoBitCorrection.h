//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/cpsr/dsp/CPSRTwoBitCorrection.h,v $
   $Revision: 1.12 $
   $Date: 2002/11/10 21:49:05 $
   $Author: wvanstra $ */

#ifndef __CPSRTwoBitCorrection_h
#define __CPSRTwoBitCorrection_h

#include "dsp/TwoBitCorrection.h"

namespace dsp {
  
  //! Converts CPSR data from 2-bit digitized to floating point values
  class CPSRTwoBitCorrection: public TwoBitCorrection {

  public:

    //! Constructor initializes base class atributes
    CPSRTwoBitCorrection ();

    ~CPSRTwoBitCorrection () { destroy(); }

    //! Return true if CPSRTwoBitCorrection can convert the Observation
    virtual bool matches (const Observation* observation);

  protected:

    //! Unpacking interface
    void unpack ();

    //! Unpacking algorithm
    void iq_unpack (float* outdata, const unsigned char* raw, 
		    uint64 ndat, unsigned channel, unsigned* weights);

    //! Temporary storage of bit-shifted values
    unsigned char* values;

    //! Build the dynamic level setting lookup table and temporary space
    void build ();

    //! Delete allocated resources
    void destroy ();

  };
  
}

#endif

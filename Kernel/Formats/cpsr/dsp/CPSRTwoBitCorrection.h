//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/cpsr/dsp/CPSRTwoBitCorrection.h,v $
   $Revision: 1.10 $
   $Date: 2002/11/06 06:30:41 $
   $Author: hknight $ */

#ifndef __CPSRTwoBitCorrection_h
#define __CPSRTwoBitCorrection_h

class CPSRTwoBitCorrection;

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
		    int64 ndat, int channel, int* weights);

    //! Temporary storage of bit-shifted values
    unsigned char* values;

    //! Build the dynamic level setting lookup table and temporary space
    void build ();

    //! Delete allocated resources
    void destroy ();

  };
  
}

#endif

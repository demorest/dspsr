//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/cpsr/dsp/CPSRTwoBitCorrection.h,v $
   $Revision: 1.6 $
   $Date: 2002/10/07 01:48:37 $
   $Author: wvanstra $ */

#ifndef __CPSRTwoBitCorrection_h
#define __CPSRTwoBitCorrection_h

#include <vector>

#include "TwoBitCorrection.h"
#include "environ.h"

namespace dsp {
  
  //! Converts a CPSR Timeseries from 2-bit digitized to floating point values
  class CPSRTwoBitCorrection: public TwoBitCorrection {

  public:

    //! Constructor
    CPSRTwoBitCorrection ();

    ~CPSRTwoBitCorrection () { destroy(); }

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

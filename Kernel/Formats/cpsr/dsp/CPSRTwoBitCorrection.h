//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/cpsr/dsp/CPSRTwoBitCorrection.h,v $
   $Revision: 1.4 $
   $Date: 2002/07/15 06:34:28 $
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
    CPSRTwoBitCorrection (int nsample = 512, float cutoff_sigma = 3.0);

    ~CPSRTwoBitCorrection () { destroy(); }

    //! Build the dynamic level setting lookup table and temporary space
    void build (int nsample, float cutoff_sigma);

  protected:

    //! Unpacking interface
    void unpack ();

    //! Unpacking algorithm
    void iq_unpack (float* outdata, const unsigned char* raw, 
		    int64 ndat, int channel, int* weights);

    //! Temporary storage of bit-shifted values
    unsigned char* values;

    //! Delete allocated resources
    void destroy ();
  };
  
}

#endif

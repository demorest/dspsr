//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/cpsr/dsp/CPSRTwoBitCorrection.h,v $
   $Revision: 1.1 $
   $Date: 2002/07/10 06:59:34 $
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

    //! Null constructor
    CPSRTwoBitCorrection (const char* name = "CPSRTwoBitCorrection",
			  Behaviour type = outofplace) :
      TwoBitCorrection (name, type) { values = 0; }

    ~CPSRTwoBitCorrection () { destroy(); }

    //! Build the dynamic level setting lookup table and temporary space
    void build (int nchannel, int nsample, float cutoff_sigma);

  protected:

    //! Unpacking interface
    void unpack ();

    //! Unpacking algorithm
    void iq_unpack (float* outdata, void* raw, unsigned long ndat, 
		    int channel, int* weights);

    //! Temporary storage of bit-shifted values
    unsigned char* values;

    //! Delete allocated resources
    void destroy ();
  };
  
}

#endif

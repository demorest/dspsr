//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/cpsr2/dsp/CPSR2TwoBitCorrection.h,v $
   $Revision: 1.1 $
   $Date: 2002/08/03 15:56:24 $
   $Author: pulsar $ */

#ifndef __CPSR2TwoBitCorrection_h
#define __CPSR2TwoBitCorrection_h

#include <vector>

#include "TwoBitCorrection.h"
#include "environ.h"

namespace dsp {
  
  //! Converts a CPSR2 Timeseries from 2-bit digitized to floating point values
  class CPSR2TwoBitCorrection: public TwoBitCorrection {

  public:

    //! Constructor
    CPSR2TwoBitCorrection (int nsample = 512, float cutoff_sigma = 3.0);

    ~CPSR2TwoBitCorrection () { destroy(); }

    //! Build the dynamic level setting lookup table and temporary space
    void build (int nsample, float cutoff_sigma);

    //! Calculate the mean voltage and power from Bit_Stream data
    virtual int64 stats (vector<double>& mean, vector<double>& power);

  protected:

    //! Unpacking interface
    void unpack ();


    //! Delete allocated resources
    void destroy ();
  };
  
}

#endif

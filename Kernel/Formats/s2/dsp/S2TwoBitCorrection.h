//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/s2/dsp/S2TwoBitCorrection.h,v $
   $Revision: 1.1 $
   $Date: 2002/10/04 10:35:41 $
   $Author: wvanstra $ */

#ifndef __S2TwoBitCorrection_h
#define __S2TwoBitCorrection_h

#include <vector>

#include "TwoBitCorrection.h"
#include "environ.h"

namespace dsp {

  class TwoBitTable;

  //! Converts a S2 Timeseries from 2-bit digitized to floating point values
  class S2TwoBitCorrection: public TwoBitCorrection {

  public:

    //! Constructor
    S2TwoBitCorrection (unsigned nsample = 512, float cutoff_sigma = 3.0);

    ~S2TwoBitCorrection () { }


  protected:

    //! Unpacking interface
    void unpack ();


  };
  
}

#endif

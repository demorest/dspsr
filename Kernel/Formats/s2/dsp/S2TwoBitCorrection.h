//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/s2/dsp/S2TwoBitCorrection.h,v $
   $Revision: 1.7 $
   $Date: 2002/11/06 06:30:42 $
   $Author: hknight $ */

#ifndef __S2TwoBitCorrection_h
#define __S2TwoBitCorrection_h

class S2TwoBitCorrection;

#include "dsp/TwoBitCorrection.h"
#include "dsp/Telescope.h"

namespace dsp {

  //! Converts S2 data from 2-bit digitized to floating point values
  class S2TwoBitCorrection: public TwoBitCorrection {

  public:

    //! Construct based on the telescope at which the data was recorded
    S2TwoBitCorrection (char telescope = Telescope::Parkes);

    //! Return true if S2TwoBitCorrection can convert the Observation
    virtual bool matches (const Observation* observation);

  protected:

    //! Unpacking interface
    void unpack ();

    //! Match the unpacking scheme to the Observation
    void match (const Observation* observation);

    //! Match the unpacking scheme to the telescope
    void match (char telescope);

    //! Interval in seconds between data resynch of S2-DAS
    double resynch_period;

    //! Start time of data resynch
    double resynch_start;

    //! End time of data resynch
    double resynch_end;

  };
  
}

#endif

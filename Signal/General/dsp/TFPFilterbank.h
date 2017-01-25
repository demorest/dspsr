//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/TFPFilterbank.h

#ifndef __TFPFilterbank_h
#define __TFPFilterbank_h

#include "dsp/Filterbank.h"

namespace dsp {
  
  //! Breaks a single-band TimeSeries into multiple frequency channels
  /*! Output will be in time, frequency, polarization order */

  class TFPFilterbank: public Filterbank {

  public:

    //! Null constructor
    TFPFilterbank ();

  protected:

    //! Perform the filterbank step 
    virtual void filterbank ();
    virtual void custom_prepare ();

  private:

    //! pscrunch flag
    unsigned pscrunch;
  };

}

#endif


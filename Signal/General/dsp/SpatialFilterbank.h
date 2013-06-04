//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2012 by Willem van Straten and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __SpatialFilterbank_h
#define __SpatialFilterbank_h

#include "dsp/Filterbank.h"

namespace dsp {
  
  //! Breaks a single-band TimeSeries into multiple frequency channels
  /*! Output will be in time, polarization, frequency order */

  class SpatialFilterbank: public Filterbank {

  public:

    //! Null constructor
    SpatialFilterbank ();

  protected:

    //! Perform the filterbank step 
    virtual void filterbank ();
    virtual void custom_prepare ();

  private:

  };

}

#endif


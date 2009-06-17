//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2007 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/LoadToFold.h,v $
   $Revision: 1.5 $
   $Date: 2009/06/17 10:16:54 $
   $Author: straten $ */

#ifndef __baseband_dsp_LoadToFold_h
#define __baseband_dsp_LoadToFold_h

#include "Reference.h"
#include "environ.h"

namespace dsp {

  class Input;

  //! Load, unpack, process and fold data into phase-averaged profile(s)
  class LoadToFold : public Reference::Able {

  public:

    //! Configuration parameters
    class Config;

    //! Set the configuration to be used in prepare and run
    virtual void set_configuration (Config*) = 0;

    //! Set the Input from which data will be read
    virtual void set_input (Input*) = 0;

    //! Prepare to fold the input TimeSeries
    virtual void prepare () = 0;

    //! Run through the data
    virtual void run () = 0;
    
    //! Finish everything
    virtual void finish () = 0;

    //! Get the minimum number of samples required to process
    virtual uint64_t get_minimum_samples () const = 0;

  };
}

#endif // !defined(__LoadToFold_h)






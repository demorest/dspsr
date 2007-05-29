//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/Attic/LoadToFold.h,v $
   $Revision: 1.1 $
   $Date: 2007/05/29 12:05:06 $
   $Author: straten $ */

#ifndef __baseband_dsp_LoadToFold_h
#define __baseband_dsp_LoadToFold_h

#include "Reference.h"

namespace dsp {

  class Input;

  //! Load, unpack, process and fold data into phase-averaged profile(s)
  class LoadToFold : public Reference::Able {

  public:

    //! Constructor
    LoadToFold ();
    
    //! Destructor
    ~LoadToFold ();

    //! Set the Input from which data will be read
    virtual void set_input (Input*) = 0;

    //! Prepare to fold the input TimeSeries
    virtual void prepare () = 0;

    //! Run through the data
    virtual void run () = 0;


  };
}

#endif // !defined(__LoadToFold_h)






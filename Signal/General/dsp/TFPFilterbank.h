//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/TFPFilterbank.h,v $
   $Revision: 1.2 $
   $Date: 2011/08/04 21:06:12 $
   $Author: straten $ */

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

    //! Engine used to perform discrete convolution step
    class Engine;
    void set_engine (Engine*);

  protected:

    //! Perform the filterbank step 
    virtual void filterbank ();
    virtual void custom_prepare ();

    //! Interface to alternate processing engine (e.g. GPU)
    Reference::To<Engine> engine;

  private:

    //! number of FFTs to scrunch in time
    unsigned tscrunch;

    //! pscrunch flag
    unsigned pscrunch;

    unsigned debugd;

  };
 
  class TFPFilterbank::Engine : public Reference::Able
  {
  public:
      Engine () {}
  }; 
}

#endif


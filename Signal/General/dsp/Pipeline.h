//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/Pipeline.h

#ifndef __dspsr_Pipeline_h
#define __dspsr_Pipeline_h

#include "Reference.h"
#include "environ.h"

namespace dsp {

  class Input;

  //! Abstract base class of data reduction pipelines

  class Pipeline : public Reference::Able
  {

  public:

    //! Set the Input from which data are read
    virtual void set_input (Input*) = 0;

    //! Get the Input from which data are read
    virtual Input* get_input () = 0;

    //! Build the data reduction pipeline
    virtual void construct () = 0;

    //! Prepare to process the input TimeSeries
    virtual void prepare () = 0;

    //! Run through the data
    virtual void run () = 0;
    
    //! Finish everything
    virtual void finish () = 0;

    //! Get the minimum number of samples required to process
    virtual uint64_t get_minimum_samples () const = 0;

  };
}

#endif // !defined(__Pipeline_h)






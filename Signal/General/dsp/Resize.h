//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2010 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __Resize_h
#define __Resize_h

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"
#include "Reference.h"

namespace dsp {

  class TimeSeries;

  //! Apply simple resize operation inplace to timeseries
  class Resize: public Transformation<TimeSeries,TimeSeries> {

  public:

    //! Default constructor
    Resize ();

    //! Destructor
    ~Resize ();

    //! Set the number of samples to be resized
    void set_resize_samples ( int64_t samples );

  protected:

    //! Perform the transformation on the input time series
    void transformation ();

    //! number of samples to adjust 
    int64_t resize_samples;

  private:

  };
  
}

#endif

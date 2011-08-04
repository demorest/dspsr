//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2010 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __RFIZapper_h
#define __RFIZapper_h

#include "dsp/TFPFilterbank.h"
#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"
#include "Reference.h"
#include "MJD.h"

namespace dsp {

  class IOManager;
  class TimeSeries;
  class TFPFilterbank;

  //! Real-time RFI zapping using a TFP Filterbank 
  class RFIZapper: public Transformation<TimeSeries,TimeSeries> {

  public:

    //! Default constructor
    RFIZapper ();

    //! Destructor
    ~RFIZapper ();

    //! Set the raw/original time series [pre filterbank]
    void set_original_input (TimeSeries * input);

    //! Get the pointer to the TFP Filterbank 
    TFPFilterbank * get_tfp_filterbank() { return tfpfilterbank; }

  protected:

    //! Perform the transformation on the input time series
    void transformation ();

    //! The buffer into which the input datastrean will be read
    Reference::To<TimeSeries> original_input;

    //! The tool for producing the TFPFilterbank
    Reference::To<TFPFilterbank> tfpfilterbank;

    //! The number of adjacent blocks to be used in SK estimator
    unsigned M;

  private:

  };
  
}

#endif

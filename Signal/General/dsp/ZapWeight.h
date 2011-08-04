//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2010 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __ZapWeight_h
#define __ZapWeight_h

#include "dsp/Transformation.h"
//#include "dsp/WeightedTimeSeries.h"
#include "dsp/TimeSeries.h"
#include "Reference.h"
#include "MJD.h"

namespace dsp {

  class IOManager;
  class TimeSeries;

  //! Apply SKFilterbank results to a timeseries
  class ZapWeight: public Transformation<TimeSeries,TimeSeries> {

  public:

    //! Default constructor
    ZapWeight ();

    //! Destructor
    ~ZapWeight ();

    //! Set the raw/original time series [pre filterbank]
    void set_skfb_input (TimeSeries * input);
    const TimeSeries* get_skfb_input () const;

    //! Set the M factor from the SKFilterbank
    void set_M (unsigned _M);

    //! Set the RFI thresholds with the specified factor
    void set_thresholds (float factor);

    //! Engine used to perform discrete convolution step
    class Engine;

    void set_engine (Engine*);

  protected:

    //! Perform the transformation on the input time series
    void transformation ();

    //! The time series containing the SKFilterbank V^2 estimates
    Reference::To<TimeSeries> skfb_input;

    //! Interface to alternate processing engine (e.g. GPU)
    Reference::To<Engine> engine;

    //! The number of adjacent blocks to be used in SK estimator
    unsigned M;

    double ddfb_rate;

    double skfb_rate;

    float sigma;

    float upper_thresh;

    float lower_thresh;

    float mega_upper_thresh;

    float mega_lower_thresh;

    uint64_t total_idats;

    uint64_t total_zaps;

    unsigned debugd;

  private:

  };
  
  class ZapWeight::Engine : public Reference::Able
  { 
  public:

    virtual void setup () = 0;

    virtual void perform (const TimeSeries* in, TimeSeries* out) = 0;

  }; 


}

#endif

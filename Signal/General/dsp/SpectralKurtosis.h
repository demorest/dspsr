//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2013 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"

#ifndef __SpectralKurtosis_h
#define __SpectralKurtosis_h

namespace dsp {
  
  //! Perform Spectral Kurtosis on Input Timeseries, creating output Time Series
  /*! Output will be in time, frequency, polarization order */

  class SpectralKurtosis: public Transformation<TimeSeries,TimeSeries> {

  public:

    //! Null constructor
    SpectralKurtosis  ();
    ~SpectralKurtosis ();

    bool get_order_supported (TimeSeries::Order order) const;

    //! Engine used to perform discrete convolution step
    class Engine;

    void set_engine (Engine*);

    void set_M (unsigned _M) { tscrunch = _M; }

    uint64_t get_skfb_inc (uint64_t blocksize);

    void set_output_tscr (TimeSeries * _output_tscr);

    void prepare ();

    void prepare_output ();

  protected:

    //! Interface to alternate processing engine (e.g. GPU)
    Reference::To<Engine> engine;

    //! Perform the transformation on the input time series
    void transformation ();

  private:

    //! number of FFTs to average to use in SK estimator
    unsigned tscrunch;

    unsigned debugd;

    //! Tsrunched SK statistic timeseries for the current block
    Reference::To<TimeSeries> output_tscr;

    //! accumulation arrays for S1 and S2 in t scrunch
    std::vector <float> S1_tscr;
    std::vector <float> S2_tscr;

  };
 
  class SpectralKurtosis::Engine : public Reference::Able
  {
  public:
      Engine () {}
  }; 
}

#endif


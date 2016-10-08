//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2010 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __SKMasker_h
#define __SKMasker_h

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"
#include "dsp/BitSeries.h"
#include "Reference.h"
#include "MJD.h"

namespace dsp {

  class IOManager;
  class TimeSeries;

  //! Apply SKFilterbank results to a timeseries
  class SKMasker: public Transformation<TimeSeries,TimeSeries> {

  public:

    //! Default constructor
    SKMasker ();

    //! Destructor
    ~SKMasker ();

    //! Set the SK Masked bitseries input
    void set_mask_input (BitSeries * input);

    //! Get the SK Masked bitseries input
    const BitSeries * get_mask_input () const;

    //! Set the M factor from the SKFilterbank
    void set_M (unsigned _M);

    //! Engine used to perform masking
    class Engine;

    void set_engine (Engine*);

  protected:

    //! Perform the transformation on the input time series
    void transformation ();

    //! The time series containing the SKFilterbank V^2 estimates
    Reference::To<BitSeries> mask_input;

    //! Interface to alternate processing engine (e.g. GPU)
    Reference::To<Engine> engine;

    //! The number of adjacent blocks to be used in SK estimator
    unsigned M;

    double ddfb_rate;

    double mask_rate;

    uint64_t total_idats;

    unsigned debugd;

  private:

  };
  
  class SKMasker::Engine : public Reference::Able
  { 
  public:
    virtual void setup () = 0;

    virtual void perform (BitSeries* mask, const TimeSeries* in, TimeSeries* out, unsigned M) = 0; 

  }; 

}

#endif

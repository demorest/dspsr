//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2016 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/TimeSeries.h"
#include "dsp/Transformation.h"

#ifndef __SKComputer_h
#define __SKComputer_h

namespace dsp {

  class SKComputer: public Transformation<TimeSeries,TimeSeries> {

  public:

    //! Null constructor
    SKComputer ();

    ~SKComputer();

    //! Engine used to perform discrete convolution step
    class Engine;

    void set_engine (Engine*);

  protected:

    //! Perform the transformation on the input time series
    void transformation ();

    //! Interface to alternate processing engine (e.g. GPU)
    Reference::To<Engine> engine;

  };

  class SKComputer::Engine : public Reference::Able
  {
  public:
      Engine () {}

      virtual void setup () = 0;

      virtual void compute (const dsp::TimeSeries* input, dsp::TimeSeries* output,
                            dsp::TimeSeries *output_tscr, unsigned tscrunch) = 0;

      virtual void insertsk (const dsp::TimeSeries* input, dsp::TimeSeries* output,
                             unsigned tscrunch) = 0;
  };
}

#endif

//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2015 by Matthew Kerr
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __FZoom_h
#define __FZoom_h

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"

namespace dsp {

  //! Perform a coarse (channelized) zoom to specified freq/bw.
  class FZoom : public Transformation <TimeSeries, TimeSeries>
  {

  public:

    FZoom ();
    
    void set_centre_frequency ( double freq );
    void set_bandwidth ( double bw );

    double get_centre_frequency ( ) const;
    double get_bandwidth (  ) const;

    //! Given an input and a goal freq / bandwidth, select channel bounds
    static void set_channel_bounds(const Observation* input,
        double centre_frequency, double bandwidth,
        unsigned* chan_lo, unsigned* chan_hi);

    class Engine;

    void set_engine (Engine*);

  protected:

    //! Determine which channels to use
    void set_bounds ();

    //! Set up output
    void prepare () ;

    //! Perform channel selection
    void transformation ();

    double centre_frequency;
    double bandwidth;

    unsigned chan_lo,chan_hi;
    void fpt_copy(TimeSeries* dest);
    void tfp_copy(TimeSeries* dest);

    Reference::To<Engine> engine;

  };

  class FZoom::Engine : public OwnStream
  {
  public:

    //! Set sense of memory copy -- device to device or device to host
    enum  Direction {

      //! Copy selected channels from device to host
      DeviceToHost,

      //! Copy selected channels from device to device
      DeviceToDevice

    };

    virtual void fpt_copy (const dsp::TimeSeries * in, 
                         dsp::TimeSeries * out,
                         unsigned chan_lo,
                         unsigned chan_hi) = 0;

    void set_direction (Direction);
    Direction get_direction () {return direction;}

  protected:

    Direction direction;

  };
}

#endif // !defined(__FZoom_h)

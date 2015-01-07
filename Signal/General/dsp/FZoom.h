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

    //! Always out of place
    FZoom ();
    
    void set_centre_frequency ( double freq );
    void set_bandwidth ( double bw );

    double get_centre_frequency ( ) const;
    double get_bandwidth (  ) const;

  protected:

    //! Determine which channels to use
    void set_bounds ();

    //! Set up output
    void prepare () ;

    //! Perform channel selection
    void transformation ();

    double centre_frequency;
    double bandwidth;

  private:
    unsigned chan_lo,chan_hi;
    void fpt_copy();
    void tfp_copy();

  };

}

#endif // !defined(__FZoom_h)

//-*-C++-*-

#ifndef __TScrunch_h
#define __TScrunch_h

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"

namespace dsp {

  //! Time-scrunches a TimeSeries by ScrunchFactor OR to a required time resolution- which ever is specified as the last setting before transformation() is called.

  class TScrunch : public Transformation <TimeSeries, TimeSeries> {

  public:

    TScrunch ();
    
    void set_ScrunchFactor( int64 _ScrunchFactor){ ScrunchFactor = _ScrunchFactor; use_tres = false; }
    int64 get_ScrunchFactor(){ return ScrunchFactor; }

    void set_NewTimeRes( double microseconds );
    /* Returns number of microseconds between time samples for output */
    double get_TimeRes(){ return TimeRes; }
    
    /* If you're not sure which method is being used, call one of these two */
    bool UsingScrunchFactor(){ return !use_tres; }
    bool UsingTimeRes(){ return use_tres; }

  protected:
    virtual void transformation ();

    int64 ScrunchFactor;
    double TimeRes;  // In microseconds

    // If true, use the tres parameter, if false use the ScrunchFactor parameter
    bool use_tres; 
    
  };

}

#endif // !defined(__TScrunch_h)

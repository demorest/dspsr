//-*-C++-*-

#ifndef __TScrunch_h
#define __TScrunch_h

#include "Operation.h"

namespace dsp {

  //! Time-scrunches a Timeseries by ScrunchFactor OR to a required time resolution- which ever is specified as the last setting before operation() is called.

  class TScrunch : public Operation {

  public:

    TScrunch(Behaviour _type);// : Operation ("TScrunch", _type);
    
    void set_ScrunchFactor( int _ScrunchFactor){ ScrunchFactor = _ScrunchFactor; use_tres = false; }
    int64 get_ScrunchFactor(){ return ScrunchFactor; }

    void set_NewTimeRes( double microseconds );
    /* Returns number of microseconds between time samples for output */
    double get_TimeRes(){ return TimeRes; }
    
    /* If you're not sure which method is being used, call one of these two */
    bool UsingScrunchFactor(){ return !use_tres; }
    bool UsingTimeRes(){ return use_tres; }

  protected:
    virtual void operation ();

    int64 ScrunchFactor;
    double TimeRes;  // In microseconds

    // If true, use the tres parameter, if false use the ScrunchFactor parameter
    bool use_tres; 
    
  };

}

#endif // !defined(__TScrunch_h)

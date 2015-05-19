//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __FScrunch_h
#define __FScrunch_h

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"

namespace dsp {

  //! Decimates a TimeSeries in the frequency domain
  class FScrunch : public Transformation <TimeSeries, TimeSeries>
  {

  public:

    FScrunch (Behaviour place=anyplace);
    
    void set_factor ( unsigned samples );
    unsigned get_factor () const;

    void set_frequency_resolution ( double Megahertz );
    double get_frequency_resolution () const;

    class Engine;

    void set_engine (Engine*);
    
  protected:

    //! Perform decimation
    void transformation ();
    void tfp_fscrunch ();
    void fpt_fscrunch ();

    mutable unsigned factor;
    mutable double frequency_resolution;

    // if true, use the frequency_resolution parameter
    // if false, use the factor parameter
    mutable bool use_fres; 

    unsigned sfactor;
    uint64_t output_nchan;

    Reference::To<Engine> engine;
  };

  class FScrunch::Engine : public OwnStream
  {
  public:

    virtual void fpt_fscrunch (const dsp::TimeSeries * in, 
                         dsp::TimeSeries * out,
                         unsigned sfactor) = 0;

  };

}

#endif // !defined(__FScrunch_h)

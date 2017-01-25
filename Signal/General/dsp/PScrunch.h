//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/PScrunch.h

#ifndef __baseband_dsp_PScrunch_h
#define __baseband_dsp_PScrunch_h

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"

#include <vector>

namespace dsp
{
  //! PScrunch all channels and polarizations
  class PScrunch : public Transformation<TimeSeries,TimeSeries>
  {

  public:

    //! Default constructor
    PScrunch ();

    //! PScrunch to zero mean and unit variance
    void transformation ();

   class Engine;

   void set_engine (Engine*);

  protected:

    Reference::To<Engine> engine;

  };

  class PScrunch::Engine : public OwnStream
  {
  public:

    virtual void setup () = 0;

    virtual void fpt_pscrunch (const dsp::TimeSeries * in,
                               dsp::TimeSeries * out) = 0;

    virtual void tfp_pscrunch (const dsp::TimeSeries* in,
                               dsp::TimeSeries* out) = 0;

   };

}

#endif

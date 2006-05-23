//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/SampleDelay.h,v $
   $Revision: 1.1 $
   $Date: 2006/05/23 22:03:21 $
   $Author: wvanstra $ */

#ifndef __baseband_dsp_SampleDelay_h
#define __baseband_dsp_SampleDelay_h

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"

#include <vector>

namespace dsp {

  class SampleDelayFunction;

  class SampleDelay : public Transformation<TimeSeries,TimeSeries> {

  public:

    //! Default constructor
    SampleDelay ();

    //! Set the delay function
    void set_function (SampleDelayFunction*);

    //! Applies the delays to the input
    void transformation ();

    //! Get the total delay (in samples)
    uint64 get_total_delay () const;

    //! Get the zero delay (in samples)
    int64 get_zero_delay () const;

  protected:

    //! The total delay (in samples)
    uint64 total_delay;

    //! The zero delay (in samples)
    int64 zero_delay;

    //! Flag set when delays have been initialized
    bool built;

    //! Initalizes the delays
    void build ();

    //! The sample delay function
    Reference::To<SampleDelayFunction> function;

  };

}

#endif

//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/SampleDelay.h,v $
   $Revision: 1.4 $
   $Date: 2009/06/17 10:16:54 $
   $Author: straten $ */

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

    //! Computes the total delay and prepares the input buffer
    void prepare ();

    //! Applies the delays to the input
    void transformation ();

    //! Get the total delay (in samples)
    uint64_t get_total_delay () const;

    //! Get the zero delay (in samples)
    int64_t get_zero_delay () const;

  protected:

    //! The total delay (in samples)
    uint64_t total_delay;

    //! The zero delay (in samples)
    int64_t zero_delay;

    //! Flag set when delays have been initialized
    bool built;

    //! Initalizes the delays
    void build ();

    //! The sample delay function
    Reference::To<SampleDelayFunction> function;

  };

}

#endif

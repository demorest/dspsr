//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2004 by Haydon Knight
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __Accumulator_h_
#define __Accumulator_h_

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"

namespace dsp {

  class Accumulator : public Transformation<TimeSeries,TimeSeries> {

  public:

    //! Default constructor
    Accumulator();

    //! Virtual destructor
    virtual ~Accumulator();

    //! Reset the output
    virtual void reset();

    //! Set the output max_samps (ndat at which full() returns true)
    void set_max_samps(uint64_t _max_samps){ max_samps = _max_samps; }

    //! Inquire the output max_samps (ndat at which full() returns true)
    uint64_t get_max_samps(){ return max_samps; }

    //! Returns true if its time to write out the buffer
    virtual bool full();

    //! Returns true if buffer is within 'close_enough' samples of being full
    virtual bool nearly_full(uint64_t close_enough);

    //! Set the maximum size of the output buffer in samples [max_samps]
    void set_max_ndat(uint64_t _max_ndat){ max_ndat = _max_ndat; }

    //! Inquire the maximum size of the output buffer in samples [0 meaning max_samps]
    uint64_t get_max_ndat(){ return max_ndat; }

    //! Choose whether to guarantee to never drop a sample
    void set_never_drop_samples(bool _never_drop_samples){ never_drop_samples = _never_drop_samples; }

    //! Inquire whether it is guaranteed that you'll never drop a sample
    bool get_never_drop_samples(){ return never_drop_samples; }

  protected:

    //! Do the work
    void transformation();

  private:

    //! Set to true on reset
    bool append;

    //! If this is set to true, the Accumulator is guaranteed to never drop a sample
    //! However, you run the risk of slow running and using lots of RAM [false]
    bool never_drop_samples;

    //! The ndat at which full() returns true
    uint64_t max_samps;

    //! The maximum size of the output buffer in samples [max_samps]
    //! If calls to operate() go above this limit one of 3 things can happen:
    //! (a) an Error is thrown (default)
    //! (b) the output buffer is expanded to encompass the append (never_drop_samples==true)
    //! (c) Samples are dropped
    uint64_t max_ndat;

  };

}

#endif


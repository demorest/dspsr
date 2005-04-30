//-*-C++-*-

#ifndef __InputBuffering_h
#define __InputBuffering_h

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"

namespace dsp {

  //! Buffers the Transformation input
  class InputBuffering : 
  public Transformation<TimeSeries,TimeSeries>::BufferingPolicy {

    public:

      //! Set the input to be buffered
      void set_input (TimeSeries* input);

      //! Set the first sample to be used from the input next time
      void set_input_next_start (uint64 next_start_sample);

      //! Perform all buffering tasks required before transformation
      void pre_transformation ();

      //! Perform all buffering tasks required after transformation
      void post_transformation ();

    protected:

      //! The next start sample
      uint64 next_start_sample;

      //! The input to be buffered
      Reference::To<TimeSeries> input;

      //! The buffer
      Reference::To<TimeSeries> buffer;

  };

}

#endif // !defined(__InputBuffering_h)

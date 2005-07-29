//-*-C++-*-

#ifndef __InputBuffering_h
#define __InputBuffering_h

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"

namespace dsp {

  //! Buffers the Transformation input
  class InputBuffering : public BufferingPolicy {

  public:
    
    //! Default constructor
    InputBuffering (HasInput<TimeSeries>* target = 0);

    //! Set the target with input TimeSeries to be buffered
    void set_target (HasInput<TimeSeries>* input);
    
    //! Set the first sample to be used from the input next time
    void set_next_start (uint64 next_start_sample);
    
    //! Perform all buffering tasks required before transformation
    void pre_transformation ();
    
    //! Perform all buffering tasks required after transformation
    void post_transformation ();
    
  protected:
    
    //! The next start sample
    uint64 next_start_sample;

    //! The requested number of samples to be reserved in the input
    uint64 requested_reserve;

    //! The target with input TimeSeries to be buffered
    HasInput<TimeSeries>* target;
    
    //! The buffer
    Reference::To<TimeSeries> buffer;
    
  };

}

#endif // !defined(__InputBuffering_h)

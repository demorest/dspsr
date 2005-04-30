#include "dsp/InputBuffering.h"

//! Set the input to be buffered
void dsp::InputBuffering::set_input (TimeSeries* _input)
{

}

//! Set the first sample to be used from the input next time
void dsp::InputBuffering::set_input_next_start (uint64 next)
{
  next_start_sample = next;
}

//! Perform all buffering tasks required before transformation
void dsp::InputBuffering::pre_transformation ()
{
#if PSEUDO
  if (buffered_data exists)       // this condition is false on first run
    input->prepend(buffered_data)
#endif
}

//! Perform all buffering tasks required after transformation
void dsp::InputBuffering::post_transformation ()
{
#if PSEUDO
  if (buffered_data required) {
    save nsamples_required in buffered_data
    input->set_prepend_buffer_size (nsamples_required);  // for next time
  }

  input->restore ();  // so that future uses of this TimeSeries do not
                         leave data in the prepend area
#endif
}



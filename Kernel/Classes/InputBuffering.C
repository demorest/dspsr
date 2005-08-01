#include "dsp/InputBuffering.h"

dsp::InputBuffering::InputBuffering (HasInput<TimeSeries>* _target)
{
  target = _target;
  next_start_sample = 0;
  requested_reserve = 0;
}

//! Set the target with input TimeSeries to be buffered
void dsp::InputBuffering::set_target (HasInput<TimeSeries>* _target)
{
  target = _target;
}

//! Set the first sample to be used from the input next time
void dsp::InputBuffering::set_next_start (uint64 next)
{
  next_start_sample = next;
}

//! Perform all buffering tasks required before transformation
void dsp::InputBuffering::pre_transformation ()
{
  if (!requested_reserve || !buffer)
    return;

  if (requested_reserve != buffer->get_ndat())
    throw Error (InvalidState, "dsp::InputBuffering::pre_transformation",
		 "requested_reserve="UI64" != buffer ndat="UI64, 
		 requested_reserve, buffer->get_ndat());

  const_cast<TimeSeries*>(target->get_input())->prepend(buffer);
}

//! Perform all buffering tasks required after transformation
void dsp::InputBuffering::post_transformation ()
{
  uint64 ndat = target->get_input()->get_ndat();

  if (next_start_sample > ndat)
    throw Error (InvalidState, "dsp::InputBuffering::post_transformation",
		 "next_start_sample="UI64" != input ndat="UI64, 
		 next_start_sample, ndat);

  int64 to_save = ndat - next_start_sample;

  target->get_input()->change_reserve (to_save - requested_reserve);

  if (!to_save)
    return;

  if (!buffer)
    buffer = new TimeSeries;

  buffer->set_nchan( target->get_input()->get_nchan() );
  buffer->set_npol ( target->get_input()->get_npol() );
  buffer->set_ndim ( target->get_input()->get_ndim() );
  buffer->resize( to_save );
  buffer->copy_data( target->get_input(), 0, to_save );
}



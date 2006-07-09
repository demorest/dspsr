/***************************************************************************
 *
 *   Copyright (C) 2005 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/InputBuffering.h"

dsp::InputBuffering::InputBuffering (HasInput<TimeSeries>* _target)
{
  target = _target;
  next_start_sample = 0;
  requested_reserve = 0;
  name = "InputBuffering";
}

//! Set the target with input TimeSeries to be buffered
void dsp::InputBuffering::set_target (HasInput<TimeSeries>* _target)
{
  target = _target;
}

/*! Copy remaining data from the target Transformation's input to buffer */
void dsp::InputBuffering::set_next_start (uint64 next)
{
  if (Operation::verbose)
    cerr << "dsp::InputBuffering::set_next_start " << next << endl;

  next_start_sample = next;

  uint64 ndat = target->get_input()->get_ndat();

  uint64 buffer_ndat = ndat - next_start_sample;

  if (next_start_sample > ndat)
    buffer_ndat = 0;

  if (Operation::verbose)
    cerr << "dsp::InputBuffering::set_next_start saving "
	 << buffer_ndat << " samples" << endl;

  if (!buffer_ndat) {
    if (buffer)
      buffer->set_ndat (0);
    return;
  }

  if (minimum_samples < buffer_ndat)
    minimum_samples = buffer_ndat;

  if (requested_reserve < minimum_samples) {
    target->get_input()->change_reserve (minimum_samples-requested_reserve);
    requested_reserve = minimum_samples;
  }

  if (!buffer)
    buffer = target->get_input()->null_clone();

  buffer->set_nchan( target->get_input()->get_nchan() );
  buffer->set_npol ( target->get_input()->get_npol() );
  buffer->set_ndim ( target->get_input()->get_ndim() );
  buffer->resize( minimum_samples );
  buffer->copy_data( target->get_input(), next_start_sample, buffer_ndat );
  buffer->set_ndat( buffer_ndat );
}

/*! Prepend buffered data to target Transformation's input TimeSeries */
void dsp::InputBuffering::pre_transformation ()
{
  if (!requested_reserve || !buffer)
    return;

  if (Operation::verbose)
    cerr << "dsp::InputBuffering::pre_transformation prepend "
	 << buffer->get_ndat() << " samples" << endl;

  const_cast<TimeSeries*>( target->get_input() )->prepend (buffer);
}

/*! No action required after transformation */
void dsp::InputBuffering::post_transformation ()
{
}



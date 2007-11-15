/***************************************************************************
 *
 *   Copyright (C) 2005 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/InputBuffering.h"

using namespace std;

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

//! Set the minimum number of samples that can be processed
void dsp::InputBuffering::set_minimum_samples (uint64 samples)
{
  minimum_samples = samples;

  if (requested_reserve < minimum_samples) {
    target->get_input()->change_reserve (minimum_samples-requested_reserve);
    requested_reserve = minimum_samples;
  }
}

/*! Copy remaining data from the target Transformation's input to buffer */
void dsp::InputBuffering::set_next_start (uint64 next)
{
  if (Operation::verbose)
    cerr << "dsp::InputBuffering::set_next_start " << next << endl;

  const TimeSeries* input = target->get_input();

  next_start_sample = next;

  // the number of samples in the target
  uint64 ndat = input->get_ndat();

  // the number of samples to be buffered
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
    set_minimum_samples (buffer_ndat);

  if (!buffer)
    buffer = input->null_clone();

  if (Operation::verbose)
    cerr << "dsp::InputBuffering::set_next_start copying from input sample "
	 << input->get_input_sample() + next_start_sample << endl;

  buffer->set_nchan( input->get_nchan() );
  buffer->set_npol ( input->get_npol() );
  buffer->set_ndim ( input->get_ndim() );
  buffer->resize( minimum_samples );
  buffer->copy_data( input, next_start_sample, buffer_ndat );
  buffer->set_ndat( buffer_ndat );
}

/*! Prepend buffered data to target Transformation's input TimeSeries */
void dsp::InputBuffering::pre_transformation ()
{
  if (!requested_reserve || !buffer || !buffer->get_ndat())
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


int64 dsp::InputBuffering::get_next_contiguous () const
{
  if (!buffer)
    return -1;

  return buffer->get_input_sample() + buffer->get_ndat();
}

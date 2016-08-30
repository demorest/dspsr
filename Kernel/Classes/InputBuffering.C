/***************************************************************************
 *
 *   Copyright (C) 2005-2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/InputBuffering.h"
#include "dsp/Reserve.h"

using namespace std;

dsp::InputBuffering::InputBuffering (HasInput<TimeSeries>* _target)
{
  target = _target;
  next_start_sample = 0;

  name = "InputBuffering";
  reserve = new Reserve;
}

//! Set the target with input TimeSeries to be buffered
void dsp::InputBuffering::set_target (HasInput<TimeSeries>* _target)
{
  target = _target;
}

//! Set the minimum number of samples that can be processed
void dsp::InputBuffering::set_minimum_samples (uint64_t samples)
{
  reserve->reserve( get_input(), samples );
}

/*! Copy remaining data from the target Transformation's input to buffer */
void dsp::InputBuffering::set_next_start (uint64_t next)
{
  const TimeSeries* input = get_input();

  next_start_sample = next;

  // the number of samples in the target
  const uint64_t ndat = input->get_ndat();

  if (Operation::verbose)
    cerr << "dsp::InputBuffering::set_next_start next=" << next 
         << " ndat=" << ndat << endl;

  if (ndat && input->get_input_sample() < 0)
    throw Error (InvalidState, "dsp::InputBuffering::set_next_start",
                 "input_sample of target input TimeSeries is not set");

  // the number of samples to be buffered
  uint64_t buffer_ndat = ndat - next_start_sample;

  if (next_start_sample > ndat)
    buffer_ndat = 0;

  if (Operation::verbose)
    cerr << "dsp::InputBuffering::set_next_start saving "
         << buffer_ndat << " samples" << endl;

  reserve->reserve( input, buffer_ndat );

  if (!buffer)
  {
    if (Operation::verbose)
      cerr << "dsp::InputBuffering::set_next_start null_clone input" << endl;
    buffer = input->null_clone();
  }

  if (Operation::verbose)
  {
    cerr << "dsp::InputBuffering::set_next_start copying from input sample "
	 << input->get_input_sample() + next_start_sample << endl;
    buffer->set_cerr (cerr);
  }

  buffer->set_nchan( input->get_nchan() );
  buffer->set_npol ( input->get_npol() );
  buffer->set_ndim ( input->get_ndim() );

  if (Operation::verbose)
    cerr << "dsp::InputBuffering::set_next_start resize buffer"
      " minimum_samples=" << reserve->get_reserved() << endl;

  buffer->resize( reserve->get_reserved() );
  buffer->copy_data( input, next_start_sample, buffer_ndat );
  buffer->set_ndat( buffer_ndat );
}

/*! Prepend buffered data to target Transformation's input TimeSeries */
void dsp::InputBuffering::pre_transformation () try
{
  if (!reserve->get_reserved() || !buffer || !buffer->get_ndat())
    return;

  const TimeSeries* container = get_input();

  int64_t want = container->get_input_sample();

  // don't wait for data preceding the first loaded block or last empty block
  if (want <= 0)
    return;

  if (buffer->get_input_sample() >= want)
    return;

  int64_t have = buffer->get_input_sample() + buffer->get_ndat();
  if (have > want)
  {
    if (Operation::verbose)
      cerr << "dsp::InputBuffering::pre_transformation buffer->get_ndat()" 
           << buffer->get_ndat() << " have=" << have << " want=" << want << endl;
    buffer->set_ndat ( buffer->get_ndat() - have + want );
  }

  if (Operation::verbose)
  {
    cerr << "dsp::InputBuffering::pre_transformation prepend "
	       << buffer->get_ndat() << " samples" << endl;
    cerr << "dsp::InputBuffering::pre_transformation target input sample="
         << want << endl;
  }

  const_cast<TimeSeries*>( container )->prepend (buffer);
}
catch (Error& error)
{
  throw error += "dsp::InputBuffering::pre_transformation";
}
/*! No action required after transformation */
void dsp::InputBuffering::post_transformation ()
{
}


int64_t dsp::InputBuffering::get_next_contiguous () const
{
  if (!buffer)
    return -1;

  return buffer->get_input_sample() + buffer->get_ndat();
}


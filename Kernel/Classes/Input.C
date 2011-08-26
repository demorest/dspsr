/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Input.h"
#include "dsp/BitSeries.h"

#include "ThreadContext.h"
#include "Error.h"

using namespace std;

dsp::Input::Input (const char* name) : Operation (name)
{
  load_size = block_size = overlap = 0;
  load_sample = 0;

  resolution = 1;
  resolution_offset = 0;

  start_offset = 0;

  last_load_ndat = 0;

  context = 0;
}

dsp::Input::~Input (){ }

void dsp::Input::set_context (ThreadContext* c)
{
  context = c;
}

std::string dsp::Input::get_prefix () const
{
  return "";
}

void dsp::Input::prepare ()
{
  if (verbose)
    cerr << "dsp::Input::prepare" << endl;

  // set the Observation information
  get_output()->Observation::operator=(info);
  get_output()->set_ndat (0);

  if (verbose)
    cerr << "dsp::Input::prepare output start_time="
	 << output->get_start_time() << endl;
}

void dsp::Input::reserve (BitSeries* buffer)
{
  uint64_t maximum_load_size = block_size;
  if (resolution > 1)
    maximum_load_size += 2 * resolution;

  if (verbose)
    cerr << "dsp::Input::reserve " << maximum_load_size << endl;

  buffer->resize (maximum_load_size);
}

void dsp::Input::reserve ()
{
  reserve (get_output());
}

//! Return the number nearest to and larger than big and divisible by small
template<typename Big, typename Small>
inline Big multiple_smaller (Big big, Small small)
{
  Big divides = big / small;
  return divides * small;
}

void dsp::Input::mark_output ()
{
  output->input_sample = load_sample - start_offset;
  output->input = this;

  output->request_offset = resolution_offset;
  output->request_ndat   = block_size;
}

//! Load data into the BitSeries specified by set_output
void dsp::Input::operation ()
{
  if (block_size < overlap)
    throw Error (InvalidState, "dsp::Input::operation", 
                 "block_size="UI64" < overlap="UI64, block_size, overlap);

  if (eod())
    throw Error (EndOfFile, "dsp::Input::operation",
		 "end of data for class '%s'",get_name().c_str());

  string reason;
  if (!info.state_is_valid (reason))
    throw Error (InvalidState, "dsp::Input::operation",
		 "invalid state: " + reason);

  if (verbose)
    cerr << "dsp::Input::operation [EXTERNAL] block_size=" << block_size
	 << " next_sample=" << load_sample+resolution_offset << endl;

  // set the Observation information
  output->Observation::operator=(info);

  // set the time as expected will result from the next call to load_data
  // note that output->start_time was set in the above call to operator=
  output->change_start_time (load_sample);

  if (verbose)
    cerr << "dsp::Input::operation [INTERNAL] load_size=" << load_size 
	 << " load_sample=" << load_sample << endl;

  if (verbose)
    cerr << "dsp::Input::operation call load_data Bit_Stream::ndat=" 
         << output->get_ndat () << endl;

  reserve ();

  load_data (output);

  // mark the input_sample and input attributes of the BitSeries
  mark_output ();

  if (verbose)
    cerr << "dsp::Input::operation load_data done"
      " load_sample=" << get_load_sample() << " name='" + get_name() + "'\n";

  int64_t to_seek = block_size - overlap;

  uint64_t available = output->get_ndat() - resolution_offset;

  if (available < block_size)
  {
    // should be the end of data
    if (!eod())
    {
      Error error (InvalidState, "dsp::Input::operation");
      error << "available=" << available << " < "
               "block_size=" << block_size << " but eod not set";
      throw error;
    }

    if (verbose)
      cerr << "dsp::Input::operation total_ndat=" << output->get_ndat()
           << " available=" << available 
           << " < block_size=" << block_size << endl;

    to_seek = available;

    uint64_t useful_ndat = multiple_smaller (output->get_ndat(), resolution);

    if (verbose)
      cerr << "dsp::Input::operation useful ndat=" << useful_ndat << endl;

    // ensure that ndat is a multiple of resolution
    output->resize ( useful_ndat );
    output->request_offset = resolution_offset;
    output->request_ndat = output->get_ndat() - resolution_offset;

    if (verbose)
      cerr << "dsp::Input::operation eod request_ndat="
           << output->request_ndat << endl;
  }

  last_load_ndat = output->get_ndat();

  if (verbose)
    cerr << "dsp::Input::operation calling seek(" << to_seek << ")" << endl;

  bool at_eod = eod();
  seek( to_seek, SEEK_CUR);
  set_eod( at_eod );

  if (verbose)
    cerr << "dsp::Input::operation exit with load_sample="
	 << load_sample <<endl;
}

//! Set the BitSeries to which data will be loaded
void dsp::Input::set_output (BitSeries* data)
{
  if (verbose)
    cerr << "dsp::Input::set_output (BitSeries* = " << data << ")" << endl;

  output = data;
}

/*! throws an exception if the output is not set.  To test
  if the output attribute is set, use Input::has_output. */
dsp::BitSeries* dsp::Input::get_output ()
{
  if (!output)
    throw Error (InvalidState, "dsp::Input::get_output", "no output set");

  return output; 
}

bool dsp::Input::has_output () const
{
  return output;
}

/*! copies the following behavioural and informational attributes:

  <UL>
  <LI> block_size
  <LI> overlap
  <LI> info
  <LI> resolution
  </UL>
*/
void dsp::Input::copy (const Input* input)
{
  set_block_size ( input->get_block_size() );
  set_overlap ( input->get_overlap() );

  info = input->info;
  resolution = input->resolution;
}

/*! 
  Sets the Observation attributes of data and load the next block of data.
  Because set_output and operate must be called separately, the
  only thread-safe interface to the Input class.
 */
void dsp::Input::load (BitSeries* data) try {

  if (verbose)
    cerr << "dsp::Input::load before lock" << endl;
  ThreadContext::Lock lock (context);
  if (verbose)
    cerr << "dsp::Input::load after lock" << endl;

  set_output( data );
  operate ();

  if (verbose)
    cerr << "dsp::Input::load exit" << endl;

 }
 catch (Error& error) {
   throw error += "dsp::Input::load (BitSeries*)";
 }

/*! 
  ensures that the load_sample attribute accomodates any extra 
  time samples required owing to time sample resolution. also 
  ensures that the load_size attribute is properly set.

  \param offset the number of time samples to offset
  \param whence from where to offset: SEEK_SET, SEEK_CUR, SEEK_END (see
  <unistd.h>)
*/
void dsp::Input::seek (int64_t offset, int whence)
{
  // the next sample required by the user
  uint64_t next_sample = load_sample + resolution_offset;

  switch (whence) {

  case SEEK_SET:
    if (offset < 0)
      throw Error (InvalidRange, "dsp::Input::seek", "SEEK_SET -ve offset");
    next_sample = offset;
    break;

  case SEEK_CUR:
    if (offset < -(int64_t)next_sample)
      throw Error (InvalidRange, "dsp::Input::seek", "SEEK_CUR -ve "
		   "offset="I64" and next_sample="I64,
		   offset,(int64_t)next_sample);
    next_sample += offset;
    break;

  case SEEK_END:
    if (!info.get_ndat())
      throw Error (InvalidState, "dsp::Input::seek", "SEEK_END unknown eod");

    if (offset < -int64_t(info.get_ndat()))
      throw Error (InvalidRange, "dsp::Input::seek", "SEEK_END -ve offset");

    next_sample = info.get_ndat() + offset;
    break;

  default:
    throw Error (InvalidParam, "dsp::Input::seek", "invalid whence");
  }

  // calculate the extra samples required owing to resolution
  resolution_offset = next_sample % resolution;
  load_sample = next_sample - resolution_offset;

  if (verbose)
    cerr << "dsp::Input::seek [INTERNAL] resolution=" << resolution 
         << " resolution_offset=" << resolution_offset 
	 << " load_sample=" << load_sample << endl;

  // ensure that the load_size attribute is properly set
  set_load_size ();
}

//! Seek to a sample close to the specified MJD
void dsp::Input::seek(MJD mjd)
{
  int misplacement = 0;

  if (mjd+1.0/info.get_rate() < info.get_start_time())
    misplacement = -1;

  if (mjd-1.0/info.get_rate() > info.get_end_time())
    misplacement = 1;

  double seek_seconds = (mjd-info.get_start_time()).in_seconds();

  if (misplacement)
  {
    string msg = "The given MJD (" + mjd.printall() + ") is ";
    if (misplacement < 0)
      msg += "before the start time";
    else
      msg += "after the end time";
    msg += "of the input data "
           "(" + info.get_start_time().printall() + "); "
           "difference is %lf seconds";

    throw Error (InvalidParam, "dsp::Input::seek", msg.c_str(), seek_seconds);
  }
 
  double seek_samples = seek_seconds*info.get_rate();
  uint64_t actual_seek = 0;

  if( seek_samples<0.0 )
    actual_seek = 0;
  else if( uint64_t(seek_samples) > info.get_ndat() )
    actual_seek = info.get_ndat();
  else
    actual_seek = uint64_t(seek_samples);

  if( verbose )
    fprintf(stderr,"dsp::Input::seek(MJD) will seek %f = "UI64" samples\n",
	    seek_samples, actual_seek);

  seek( actual_seek, SEEK_SET );
}

void dsp::Input::seek_seconds (double seconds, int whence)
{
  if (info.get_rate() == 0)
    throw Error (InvalidState, "dsp::Input::seek_seconds",
		 "data rate unknown");

  seek( int64_t(seconds * info.get_rate()), whence );
}

double dsp::Input::tell_seconds () const
{
  if (info.get_rate() == 0)
    throw Error (InvalidState, "dsp::Input::tell_seconds",
		 "data rate unknown");

  return load_sample / info.get_rate();
}

void dsp::Input::set_start_seconds (double seconds)
{
  if (seconds < 0)
    throw Error (InvalidParam, "dsp::Input::set_start_seconds",
		 "seconds = %lf < 0", seconds);

  seek_seconds (seconds);

  // the next sample required by the user
  start_offset = load_sample + resolution_offset;
}

//! Convenience method used to set the number of seconds
void dsp::Input::set_total_seconds (double seconds)
{
  if (seconds < 0)
    throw Error (InvalidParam, "dsp::Input::set_total_seconds",
		 "seconds = %lf < 0", seconds);

  if (info.get_rate() == 0)
    throw Error (InvalidState, "dsp::Input::set_total_seconds",
		 "data rate unknown");

  uint64_t total_samples = uint64_t( seconds * info.get_rate() );

  if ((total_samples > get_total_samples()) && (get_total_samples() > 0) )
    throw Error (InvalidParam, "dsp::Input::set_total_seconds",
		 "samples="UI64" > total samples="UI64, total_samples,
		 get_total_samples());

  set_total_samples ( total_samples );
}



/*! ensures that the load_size attribute is properly set. */
void dsp::Input::set_block_size (uint64_t size)
{
  block_size = size;
  set_load_size ();
}

/*! ensures that the load_size attribute is large enough to load
  the number of time samples requested by Input::set_block_size, as well
  as any extra time samples required owing to time sample resolution and
  the next time sample requested by Input::seek. */
void dsp::Input::set_load_size ()
{
  if (verbose)
    cerr << "dsp::Input::set_load_size block_size=" << block_size 
         << " resolution_offset=" << resolution_offset << endl;

  load_size = block_size + resolution_offset;

  uint64_t remainder = load_size % resolution;

  if (remainder)
    load_size += resolution - remainder;

  if (info.get_ndat() && load_sample + load_size > info.get_ndat())
  {
    if (verbose)
      cerr << "dsp::Input::set_load_size near end of data" << endl;

    load_size = info.get_ndat() - load_sample;
  }

  if (verbose)
    cerr << "dsp::Input::set_load_size load_size=" << load_size << endl;
}


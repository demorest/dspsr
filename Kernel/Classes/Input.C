#include <iostream>

#include "dsp/Input.h"
#include "dsp/BitSeries.h"

#include "MJD.h"
#include "environ.h"

#include "Error.h"
#include "genutil.h"

dsp::Input::Input (const char* name) : Operation (name)
{
  load_size = block_size = overlap = 0;
  load_sample = 0;

  resolution = 1;
  resolution_offset = 0;
}

dsp::Input::~Input ()
{
}

//! Load data into the BitSeries specified by set_output
void dsp::Input::operation ()
{
  if (verbose)
    cerr << "dsp::Input::operate" << endl;
  
  if (!output)
    throw Error (InvalidState, "dsp::Input::operate", "no output BitSeries");

  load (output);
}

//! Set the BitSeries to which data will be loaded
void dsp::Input::set_output (BitSeries* data)
{
  if (verbose)
    cerr << "dsp::Input::set_output (BitSeries* = " << data << ")" << endl;

  if (!output || output != data) {
    output = data;
    output -> input_sample = -1;
  }
}

/*! This method throws an exception if the output is not set.  To test
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

/*! This method copies the following behavioural and informational attributes:

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

/*! Set the Observation attributes of data and load the next block of data
 */
void dsp::Input::load (BitSeries* data)
{
  if( data!=output.ptr() ){
    if( timekeeper )
      timekeeper->setup_done();
    if (record_time)
      optime.start();
  }

  //if( verbose ) fprintf(stderr,"in dsp::Input::load() with load_size="UI64"\n",load_size);

  if (verbose)
    cerr << "dsp::Input::load (BitSeries* = " << data << ")" << endl;

  if (!data)
    throw Error (InvalidParam, "dsp::Input::load", "invalid data reference");

  if (block_size < overlap)
    throw Error (InvalidState, "dsp::Input::load", 
                 "block_size="UI64" < overlap="UI64, block_size, overlap);

  if (eod())
    throw Error (InvalidState, "dsp::Input::load", "end of data");

  string reason;
  if (!info.state_is_valid (reason))
    throw Error (InvalidState, "dsp::Input::load", "invalid state: "+reason);

  if (verbose)
    cerr << "dsp::Input::load [EXTERNAL] block_size=" << block_size
	 << " next_sample=" << load_sample+resolution_offset
         << " (overlap=" << overlap << ")" << endl;

  // set the Observation information
  data->Observation::operator=(info);

  //fprintf(stderr,"in dsp::Input::load() and have set data->ndat="UI64"\n",
  // data->get_ndat());

  // set the time as expected will result from the next call to load_data
  // note that data->start_time was set in the above call to operator=
  data->change_start_time (load_sample);

  if (verbose)
    cerr << "dsp::Input::load [INTERNAL] load_size=" << load_size 
	 << " load_sample=" << load_sample << endl;

  //fprintf(stderr,"in dsp::Input::load() and calling data->resize("UI64")\n",
  //  load_size);
  data->resize (load_size);
  //fprintf(stderr,"in dsp::Input::load() and called data->resize("UI64") to give data->get_ndat()="UI64"\n",
  //  load_size,data->get_ndat());

  if (verbose)
    cerr << "dsp::Input::load call load_data" << endl;

  load_data (data);

  data->input_sample = load_sample;
  data->request_offset = resolution_offset;

  uint64 available = data->get_ndat() - resolution_offset;
  if (available < block_size)
    data->request_ndat = available;
  else
    data->request_ndat = block_size;

  if( verbose )
    fprintf(stderr,"dsp::Input::load() calling seek("I64")\n",
	    int64(data->get_ndat()) - int64(overlap));
  seek( int64(data->get_ndat()) - int64(overlap), SEEK_CUR);
  //seek (block_size - overlap, SEEK_CUR);

  if (verbose)
    cerr << "dsp::Input::load exit with load_sample="<< load_sample <<endl;

  if( data!=output.ptr() ){
    if (record_time)
      optime.stop();
  }
}

/*! 
  This method ensures that the load_sample attribute accomodates any extra 
  time samples required owing to time sample resolution. This method also 
  ensures that the load_size attribute is properly set.

  \param offset the number of time samples to offset
  \param whence from where to offset: SEEK_SET, SEEK_CUR, SEEK_END (see
  <unistd.h>)
*/
void dsp::Input::seek (int64 offset, int whence)
{
  if (verbose)
    cerr << "dsp::Input::seek [EXTERNAL] offset=" << offset << endl;

  // the next sample required by the user
  uint64 next_sample = load_sample + resolution_offset;

  switch (whence) {

  case SEEK_SET:
    if (offset < 0)
      throw Error (InvalidRange, "dsp::Input::seek", "SEEK_SET -ve offset");
    next_sample = offset;
    break;

  case SEEK_CUR:
    if (offset < -(int64)next_sample)
      throw Error (InvalidRange, "dsp::Input::seek", "SEEK_CUR -ve "
		   "offset="I64" and next_sample="I64,
		   offset,(int64)next_sample);
    next_sample += offset;
    break;

  case SEEK_END:
    if (!info.get_ndat())
      throw Error (InvalidState, "dsp::Input::seek", "SEEK_END unknown eod");

    if (offset < -int64(info.get_ndat()))
      throw Error (InvalidRange, "dsp::Input::seek", "SEEK_END -ve offset");

    if (offset != 0)
      set_eod( false );

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
  if( mjd+1.0/info.get_rate() < info.get_start_time() )
    throw Error(InvalidParam, "dsp::Input::seek",
		"The given MJD (%s) is before the start time of the input data (%s)  (Difference is %s)",
		mjd.printall(),info.get_start_time().printall(),
		(info.get_start_time()-mjd).printall());

  if( mjd-1.0/info.get_rate() > info.get_end_time() )
    throw Error(InvalidParam,"dsp::Input::seek",
		"The given MJD (%s) is after the end time of the input data (%s)  (Difference is %f seconds)",
		mjd.printall(),info.get_start_time().printall(),
		(mjd-info.get_start_time()).in_seconds());
  
  double seek_samples = (mjd-info.get_start_time()).in_seconds()*info.get_rate();
  uint64 actual_seek = 0;

  if( seek_samples<0.0 )
    actual_seek = 0;
  else if( uint64(seek_samples) > info.get_ndat() )
    actual_seek = info.get_ndat();
  else
    actual_seek = uint64(seek_samples);

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

  seek( int64(seconds * info.get_rate()), whence );
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

  uint64 total_samples = uint64( seconds * info.get_rate() );

  if (total_samples > get_total_samples ())
    throw Error (InvalidParam, "dsp::Input::set_total_seconds",
		 "samples="UI64" > total samples="UI64, total_samples,
		 get_total_samples());

  set_total_samples ( total_samples );
}



/*! This method also ensures that the load_size attribute is properly set. */
void dsp::Input::set_block_size (uint64 size)
{
  block_size = size;
  set_load_size ();
}

/*! This method ensures that the load_size attribute is large enough to load
  the number of time samples requested by Input::set_block_size, as well
  as any extra time samples required owing to time sample resolution and
  the next time sample requested by Input::seek. */
void dsp::Input::set_load_size ()
{
  if (verbose)
    cerr << "dsp::Input::set_load_size block_size=" << block_size 
         << "resolution_offset=" << resolution_offset << endl;

  load_size = block_size + resolution_offset;

  uint64 remainder = load_size % resolution;

  if (remainder)
    load_size += resolution - remainder;

  if (verbose)
    cerr << "dsp::Input::set_load_size load_size=" << load_size << endl;
}







#include <iostream>

#include "dsp/Seekable.h"
#include "dsp/BitSeries.h"

#include "environ.h"
#include "Error.h"

//! Constructor
dsp::Seekable::Seekable (const char* name) : Input (name)
{ 
  init(); 
}
    
/*! The destructor is defined in the .C file so that the
    Reference::To<BitStream> destructor need not know about the BitStream
    class in the .h file, allowing changes to be made to BitStream without
    forcing the re-compilation of code that uses Input but does not use
    BitStream.
*/
dsp::Seekable::~Seekable ()
{
}

void dsp::Seekable::init()
{
  end_of_data = true;
  current_sample = 0;
}

void dsp::Seekable::reset ()
{
  end_of_data = false;
  current_sample = 0;
  seek (0);
  last_load_ndat = 0;
}

bool dsp::Seekable::eod()
{
  return end_of_data;
}

void dsp::Seekable::load_data (BitSeries* data)
{
  if (verbose)
    cerr << "dsp::Seekable::load_data"
      "\n   load_size=" << get_load_size() << 
      "\n   load_sample=" << get_load_sample() <<
      "\n   current_sample=" << current_sample << endl;

  uint64 recycled = recycle_data (data);

  uint64 read_sample = get_load_sample() + recycled;

  uint64 read_size = get_load_size() - recycled;

  if (verbose)
    cerr << "dsp::Seekable::load_data"
		 "\n   recycled=" << recycled << 
		 "\n   read_size=" << read_size << 
		 "\n   read_sample=" << read_sample << endl;

  // check that the amount to read does not surpass the end of data
  if (info.get_ndat()) {

    if (verbose)
      cerr << "   total ndat=" << info.get_ndat() 
	   << " read_sample=" << read_sample << endl;

    if( read_sample > info.get_ndat() )
      throw Error(InvalidState,"dsp::Seekable::load_data ()",
		  "'read_sample' > ndat.... BUG!");

    uint64 samples_left = info.get_ndat() - read_sample;

    if (verbose)
      cerr << "dsp::Seekable::load_data " << samples_left 
	   << " samples remaining" << endl;

    if (samples_left <= read_size) {

      if (verbose)
	cerr << "dsp::Seekable::load_data end of data read_size="
	     << samples_left << endl;

      read_size = samples_left;
      end_of_data = true;

    }

  }

  // exit if there is nothing left to read
  if (!read_size){
    data->set_ndat (recycled);
    return;
  }

  if (read_sample != current_sample) {

    uint64 toseek_bytes = data->get_nbytes (read_sample);

    if (verbose)
      cerr << "dsp::Seekable::load_data"
	" call seek_bytes("<< toseek_bytes <<")"<<endl;

    int64 seeked = seek_bytes (toseek_bytes);
    if (seeked < 0)
      throw Error (FailedCall, "dsp::Seekable::load_data", "error seek_bytes");
    
    // confirm that we be where we expect we be
    if (read_sample != (uint64) data->get_nsamples (seeked))
      throw Error (InvalidState, "dsp::Seekable::load_data", "seek mismatch"
		   " read_sample="UI64" absolute_sample="UI64,
		   read_sample, data->get_nsamples (seeked));

    current_sample = read_sample;
  }

  uint64 toread_bytes = data->get_nbytes (read_size);
  unsigned char* into = data->get_rawptr() + data->get_nbytes (recycled);

  if (toread_bytes < 1)
    throw Error (InvalidState, "dsp::Seekable::load_data",
		 "invalid BitSeries state");

  if (verbose)
    cerr << "dsp::Seekable::load_data"
      " call load_bytes("<< toread_bytes << ")" <<endl;

  int64 bytes_read = load_bytes (into, toread_bytes);

  if (bytes_read < 0)
    throw Error (FailedCall, "dsp::Seekable::load_data", "load_bytes error- possibly your blocksize may be too large.  (blocksize="UI64")",
		 get_block_size());

  if ((uint64)bytes_read < toread_bytes) {
    if (verbose)
      cerr << "dsp::Seekable::load_data end of data bytes_read=" << bytes_read
           << " < bytes_toread=" << toread_bytes << endl;
    end_of_data = true;
    read_size = data->get_nsamples (bytes_read);
  }

  current_sample += read_size;

  data->set_ndat (recycled + read_size);

  // verbose = false;
}

/*!  Based on the next time sample, get_load_sample, and the number of
  time samples, get_load_size, to be loaded, this function determines
  the amount of requested data that is currently found in the output
  BitSeries instance.  This data is then copied to the start of
  BitSeries::data and the number of time samples that have been
  "recycled" is returned. */
uint64 dsp::Seekable::recycle_data (BitSeries* data)
{
  if (data->get_input_sample(this) == -1)  {
    if (verbose)
      cerr << "dsp::Seekable::recycle_data no input_sample" << endl;
    return 0;
  }

  uint64 start_sample = (uint64) data->get_input_sample();
  uint64 last_sample = start_sample + last_load_ndat;

  if (verbose)
    cerr << "dsp::Seekable::recycle_data"
      "\n start_sample=" << start_sample <<
      "\n last_sample=" << last_sample << 
      "\n load_sample=" << get_load_sample() << endl;

  if (get_load_sample() < start_sample || get_load_sample() >= last_sample)
    return 0;

  uint64 to_recycle = last_sample - get_load_sample();

  if (verbose)
    cerr << "dsp::Seekable::recycle_data recycle " 
	 << to_recycle << " samples" << endl;

  if (to_recycle > get_load_size())
    to_recycle = get_load_size();

  uint64 recycle_bytes = data->get_nbytes (to_recycle);
  uint64 offset_bytes = data->get_nbytes (get_load_sample() - start_sample);

  if (verbose)
    cerr << "dsp::Seekable::recycle_data recycle " << recycle_bytes
	 << " bytes (offset=" << offset_bytes << " bytes)" << endl;

  // check if the next sample is already the start sample
  if (!offset_bytes)
    return to_recycle;

  unsigned char *into = data->get_rawptr();
  unsigned char *from = data->get_rawptr() + offset_bytes;

  // this loop is "overlap safe"
  while (recycle_bytes) {

    if (offset_bytes > recycle_bytes)
      offset_bytes = recycle_bytes;

    memcpy (into, from, offset_bytes);

    recycle_bytes -= offset_bytes;
    into += offset_bytes;
    from += offset_bytes;
  }

  if (verbose)
    cerr << "dsp::Seekable::recycle_data recycled " << to_recycle << endl;

  return to_recycle;
}


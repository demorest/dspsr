/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Seekable.h"
#include "dsp/BitSeries.h"

#include <inttypes.h>
#include "Error.h"

#include <string.h>

using namespace std;

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

void dsp::Seekable::rewind ()
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

  uint64_t recycled = recycle_data (data);

  uint64_t read_sample = get_load_sample() + recycled;

  uint64_t read_size = get_load_size() - recycled;
  
  if (verbose)
    cerr << "dsp::Seekable::load_data"
      "\n   recycled=" << recycled << 
      "\n   read_size=" << read_size << 
      "\n   read_sample=" << read_sample << endl;

  // check that the amount to read does not surpass the end of data
  if (info.get_ndat())
  {
    if (verbose)
      cerr << "dsp::Seekable::load_data total ndat=" << info.get_ndat() 
	   << " read_sample=" << read_sample << endl;

    if (read_sample > info.get_ndat())
      throw Error (InvalidState, "dsp::Seekable::load_data",
		   "read_sample="UI64" > ndat="UI64 "\n\t"
		   "recycled="UI64" load_sample="UI64,
		   read_sample, info.get_ndat(),
		   recycled, get_load_sample());

    uint64_t samples_left = info.get_ndat() - read_sample;

    if (verbose)
      cerr << "dsp::Seekable::load_data " << samples_left 
	   << " samples remaining" << endl;

    if (samples_left <= read_size)
    {
      if (verbose)
	cerr << "dsp::Seekable::load_data end of data read_size="
	     << samples_left << endl;

      read_size = samples_left;
      end_of_data = true;
    }
  }

  // exit if there is nothing left to read
  if (!read_size)
  {
    data->set_ndat (recycled);
    return;
  }

  if (read_sample != current_sample)
  {
    uint64_t toseek_bytes = data->get_nbytes (read_sample);

    if (verbose)
      cerr << "dsp::Seekable::load_data read_sample=" << read_sample
           << " != current_sample=" << current_sample 
	   << " seek_bytes=" << toseek_bytes << endl;

    int64_t seeked = seek_bytes (toseek_bytes);
    if (seeked < 0)
      throw Error (FailedCall, "dsp::Seekable::load_data", "error seek_bytes");
    
    // confirm that we be where we expect we be
    if (read_sample != (uint64_t) data->get_nsamples (seeked))
      throw Error (InvalidState, "dsp::Seekable::load_data", "seek mismatch"
		   " read_sample="UI64" absolute_sample="UI64,
		   read_sample, data->get_nsamples (seeked));

    current_sample = read_sample;
  }

  uint64_t toread_bytes = data->get_nbytes (read_size);
  unsigned char* into = data->get_rawptr() + data->get_nbytes (recycled);

  if (toread_bytes < 1)
    throw Error (InvalidState, "dsp::Seekable::load_data",
		 "invalid BitSeries state");

  if (verbose)
    cerr << "dsp::Seekable::load_data"
      " call load_bytes("<< toread_bytes << ")" <<endl;

  int64_t bytes_read = load_bytes (into, toread_bytes);

  if (bytes_read < 0)
    throw Error (FailedCall, "dsp::Seekable::load_data",
		 "load_bytes ("UI64") block_size=", toread_bytes,
		 get_block_size());

  if ((uint64_t)bytes_read < toread_bytes)
  {
    if (verbose)
      cerr << "dsp::Seekable::load_data end of data bytes_read=" << bytes_read
           << " < bytes_toread=" << toread_bytes << endl;
    end_of_data = true;
    read_size = data->get_nsamples (bytes_read);
  }

  current_sample += read_size;

  data->set_ndat (recycled + read_size);

  if (get_overlap() && overlap_buffer && !end_of_data)
  {
    uint64_t to_copy = get_overlap();
    uint64_t remainder = to_copy % resolution;
    if (remainder)
      to_copy += resolution - remainder;

    if (verbose)
      cerr << "dsp::Seekable::load_data overlap=" << get_overlap()
	   << " to_copy=" << to_copy << endl;

    overlap_buffer->set_nchan( data->get_nchan() );
    overlap_buffer->set_npol ( data->get_npol() );
    overlap_buffer->set_ndim ( data->get_ndim() );
    overlap_buffer->set_nbit ( data->get_nbit() );

    // resize to the potential maximum size
    overlap_buffer->resize( get_overlap()+resolution );

    mark_output ();

    overlap_buffer->copy_data( data, data->get_ndat()-to_copy, to_copy );
    overlap_buffer->set_ndat( to_copy );

    if (verbose)
      cerr << "dsp::Seekable::load_data overlap buffer input_sample="
	   << overlap_buffer->get_input_sample () << endl;
  }
}

/*!  Based on the next time sample, get_load_sample, and the number of
  time samples, get_load_size, to be loaded, this function determines
  the amount of requested data that is currently found in the output
  BitSeries instance.  This data is then copied to the start of
  BitSeries::data and the number of time samples that have been
  "recycled" is returned. */
uint64_t dsp::Seekable::recycle_data (BitSeries* data)
{
  BitSeries* from = data;

  if (overlap_buffer)
  {
    if (verbose)
      cerr << "dsp::Seekable::recycle_data using overlap buffer" << endl;

    from = overlap_buffer;
  }

  if (from->get_input_sample (this) == -1)
  {
    if (verbose)
      cerr << "dsp::Seekable::recycle_data no input_sample" << endl;
    return 0;
  }

  uint64_t start_sample = (uint64_t) from->get_input_sample();
  uint64_t last_sample = start_sample + last_load_ndat;

  if (overlap_buffer)
    last_sample = start_sample + overlap_buffer->get_ndat();

  if (verbose)
    cerr << "dsp::Seekable::recycle_data"
      "\n   start_sample=" << start_sample <<
      "\n   last_sample=" << last_sample << endl;

  if (get_load_sample() < start_sample || get_load_sample() >= last_sample)
    return 0;

  uint64_t to_recycle = last_sample - get_load_sample();

  if (verbose)
    cerr << "dsp::Seekable::recycle_data recycle " 
	 << to_recycle << " samples" << endl;

  if (to_recycle > get_load_size())
    to_recycle = get_load_size();

  uint64_t recycle_bytes = from->get_nbytes (to_recycle);
  uint64_t offset_bytes = from->get_nbytes (get_load_sample() - start_sample);

  if (verbose)
    cerr << "dsp::Seekable::recycle_data recycle " << recycle_bytes
	 << " bytes (offset=" << offset_bytes << " bytes)" << endl;

  unsigned char *into = data->get_rawptr();
  unsigned char *rbuf = from->get_rawptr() + offset_bytes;

  if (overlap_buffer)
    memcpy (into, rbuf, size_t(recycle_bytes));
  else
  {
    // perform an "overlap safe" memcpy

    // check if the next sample is already the start sample
    if (!offset_bytes)
      return to_recycle;
    
    while (recycle_bytes)
    {
      if (offset_bytes > recycle_bytes)
	offset_bytes = recycle_bytes;

      memcpy (into, rbuf, size_t(offset_bytes));
      
      recycle_bytes -= offset_bytes;
      into += offset_bytes;
      rbuf += offset_bytes;
    }
  }

  if (verbose)
    cerr << "dsp::Seekable::recycle_data recycled " << to_recycle << endl;

  return to_recycle;
}


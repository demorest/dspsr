#include "dsp/Seekable.h"
#include "dsp/BitSeries.h"

#include "genutil.h"

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
  Input::seek (0);
}

bool dsp::Seekable::eod()
{
  return end_of_data;
}

void dsp::Seekable::load_data (BitSeries* data)
{
  if (verbose)
    cerr << "Seekable::load_data"
      " block_size=" << block_size << 
      " next_sample=" << get_next_sample() <<
      " current_sample=" << current_sample << endl;

  uint64 recycled = recycle_data (data);

  uint64 read_sample = get_next_sample() + recycled;

  if (verbose)
    cerr << "Seekable::load_data recycled="
	 << recycled << endl;

  uint64 read_size = block_size - recycled;

  // check that the amount to read does not surpass the end of data
  if (info.get_ndat()) {
    uint64 samples_left = info.get_ndat() - read_sample;
    if (samples_left <= read_size) {
      if (verbose)
	cerr << "Seekable::load_data end of data read_size="
	     << samples_left << endl;
      read_size = samples_left;
      end_of_data = true;
    }
  }

  // exit if there is nothing left to read
  if (!read_size)
    return;

  // If current_sample == 0, ensure that we seek past the header!
  if (current_sample == 0 || read_sample != current_sample) {

    uint64 toseek_bytes = data->nbytes (read_sample);

    if (verbose)
      cerr<<"Seekable::load_data call seek_bytes("<< toseek_bytes <<")"<<endl;

    int64 seeked = seek_bytes (toseek_bytes);
    if (seeked < 0)
      throw_str ("Seekable::load_data error seek_bytes");

    // confirm that we be where we expect we be
    if (read_sample != (uint64) data->nsamples (seeked))
      throw_str ("Seekable::load_data seek mismatch"
		 " read_sample="UI64" absolute_sample="UI64,
		 read_sample, data->nsamples (seeked));

    current_sample = read_sample;
  }

  uint64 toread_bytes = data->nbytes (read_size);
  unsigned char* into = data->get_rawptr() + data->nbytes (recycled);

  if (toread_bytes < 1)
    throw_str ("Seekable::load_data invalid BitSeries state");

  if (verbose)
    cerr<<"Seekable::load_data call load_bytes("<< toread_bytes << ")" <<endl;

  int64 bytes_read = load_bytes (into, toread_bytes);

  if (bytes_read < 0)
    throw_str ("Seekable::load_data load_bytes error");

  if ((uint64)bytes_read < toread_bytes) {
    end_of_data = true;
    read_size = data->nsamples (bytes_read);
  }

  current_sample += read_size;

  data->set_ndat (recycled + read_size);
}

/*!  Based on the next time sample, next_sample, and the number of
  time samples, block_size, to be loaded, this function determines the
  amount of required data currently found in the BitSeries object,
  copies this data to the start of BitSeries::data and returns the
  number of time samples that have been "recycled" */
uint64 dsp::Seekable::recycle_data (BitSeries* data)
{
  if (data->get_input_sample() == -1)  {
    if (verbose)
      cerr << "dsp::Seekable::recycle_data no input_sample" << endl;
    return 0;
  }

  uint64 start_sample = (uint64) data->get_input_sample();
  uint64 last_sample = start_sample + (uint64) data->get_ndat();

  if (verbose)
    cerr << "dsp::Seekable::recycle_data"
      " start_sample=" << start_sample <<
      " last_sample=" << last_sample << 
      " next_sample=" << next_sample << endl;

  if (next_sample < start_sample || next_sample >= last_sample)
    return 0;

  uint64 to_recycle = last_sample - next_sample;

  if (verbose)
    cerr << "dsp::Seekable::recycle_data recycle " << to_recycle << " samples" << endl;

  if (to_recycle > block_size)
    to_recycle = block_size;

  uint64 recycle_bytes = data->nbytes (to_recycle);
  uint64 offset_bytes = data->nbytes (next_sample - start_sample);

  // next_sample += to_recycle;

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

#include "Seekable.h"
#include "Timeseries.h"
#include "genutil.h"

void dsp::Seekable::init()
{
  end_of_data = true;
  total_size = 0;
  current_sample = 0;
}

void dsp::Seekable::reset ()
{
  end_of_data = false;
  current_sample = 0;
  next_sample = 0;
}

bool dsp::Seekable::eod()
{
  return end_of_data;
}

void dsp::Seekable::load_data (Timeseries* data)
{
  if (verbose)
    cerr << "Seekable::load_block"
      " block_size=" << block_size << 
      " next_sample=" << next_sample <<
      " current_sample=" << current_sample << endl;

  uint64 recycled = recycle_data (data);

  if (verbose)
    cerr << "Seekable::load_block recycled="
	 << recycled << endl;

  uint64 read_size = block_size - recycled;

  // check that the amount to read does not surpass the end of data
  if (total_size) {
    uint64 samples_left = total_size - next_sample;
    if (samples_left <= read_size) {
      if (verbose)
	cerr << "Seekable::load_block end of data read_size="
	     << samples_left << endl;
      read_size = samples_left;
      end_of_data = true;
    }
  }

  // exit if there is nothing left to read
  if (!read_size)
    return;

  if (next_sample - current_sample) {

    uint64 toseek_bytes = data->nbytes (next_sample);

    if (verbose)
      cerr << "Seekable::load_block seek nbytes="
	   << toseek_bytes << endl;

    int64 seeked = seek_bytes (toseek_bytes);
    if (seeked < 0)
      throw_str ("Seekable::load_block error seek_bytes");

    // confirm that we be where we expect we be
    if (next_sample != (uint64) data->nsamples (seeked))
      throw_str ("Seekable::load_block seek mismatch"
		 " next_sample="UI64" absolute_sample="UI64,
		 next_sample, data->nsamples (seeked));

    current_sample = next_sample;
  }

  uint64 to_read = data->nbytes (read_size);
  unsigned char* into = data->get_rawptr() + data->nbytes (recycled);

  if (to_read < 1)
    throw_str ("Seekable::load_block invalid Timeseries state");

  if (verbose)
    cerr << "Seekable::load_block read nbytes=" << to_read << endl;

  int64 bytes_read = load_bytes (into, to_read);

  if (bytes_read < 0)
    throw_str ("Seekable::load_block load_bytes error");

  if ((uint64)bytes_read < to_read) {
    end_of_data = true;
    read_size = data->nsamples (bytes_read);
  }

  current_sample += read_size;

  data->set_ndat (recycled + read_size);

  if (overlap > read_size)
    next_sample -= (overlap - read_size);
  else
    next_sample += (read_size - overlap);
}

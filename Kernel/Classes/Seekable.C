#include "dsp/Seekable.h"
#include "dsp/Chronoseries.h"
#include "genutil.h"

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

void dsp::Seekable::load_data (Chronoseries* data)
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
    throw_str ("Seekable::load_data invalid Chronoseries state");

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

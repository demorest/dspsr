#include "Input.h"
#include "Timeseries.h"
#include "Error.h"

#include "genutil.h"

bool dsp::Input::verbose = false;

void dsp::Input::init()
{
  block_size = overlap = 0;
  next_sample = 0;
}

/*! Set the Observation attributes of data and load the next block of data
 */
void dsp::Input::load (Timeseries* data)
{
  if (!data)
    throw Error (InvalidParam, "Input::load", "invalid data reference");

  if (block_size < overlap)
    throw Error (InvalidState, "Input::load", 
                 "block_size="UI64" < overlap="UI64, block_size, overlap);

  if (eod())
    throw Error (InvalidState, "Input::load", "end of data");

  string reason;
  if (!info.state_is_valid (reason))
    throw_str ("Input::load invalid state: "+reason);

  if (verbose)
    cerr << "Input::load block_size=" << block_size
         << " overlap=" << overlap << " next=" << next_sample << endl;

  // set the Observation information
  data->Observation::operator=(info);

  // set the time as expected will result from the next call to load_data
  // note that data->start_time was set in the above call to operator=
  data->change_start_time (next_sample);

  if (verbose)
    cerr << "Input::load resize data" << endl;

  data->resize (block_size);

  load_time.start();

  if (verbose)
    cerr << "Input::load call load_data" << endl;

  load_data (data);

  load_time.stop();

  data->input_sample = next_sample;

  next_sample += block_size - overlap;

  if (verbose)
    cerr << "Input::load exit with next_sample="<< next_sample <<endl;
}

/*!
  Set from where the next "load_block" will load
  \param offset the number of time samples to offset
  \param whence from where to offset; 0==absolute, 1=relative
  \return 0 on success; -1 on failure
*/
void dsp::Input::seek (int64 offset, int whence)
{
  if (verbose)
    cerr << "Input::seek offset=" << offset << endl;

  switch (whence) {
  case 0:
    if (offset < 0)
      throw_str ("Input::seek negative offset");
    next_sample = offset;
    break;
  case 1:
    if (offset < -(int64)next_sample)
      throw_str ("Input::seek negative offset");
    next_sample += offset;
    break;
  default:
    throw_str ("Input::seek cannot seek past end of file");
  }
}

/*!  
  Based on the next time sample, next_sample, and the number of time
  samples, block_size, to be loaded, this function determines the
  amount of data currently found in the Timeseries object, copies this
  data to the start of Timeseries::data and returns the number of time
  samples that have been "recycled"
*/

uint64 dsp::Input::recycle_data (Timeseries* data)
{
  if (data->input_sample == -1)  {
    if (verbose)
      cerr << "Input::recycle_data no input_sample" << endl;
    return 0;
  }

  uint64 start_sample = (uint64) data->input_sample;
  uint64 last_sample = start_sample + (uint64) data->get_ndat();

  if (verbose)
    cerr << "Input::recycle_data"
      " start_sample=" << start_sample <<
      " last_sample=" << last_sample << 
      " next_sample=" << next_sample << endl;

  if (next_sample < start_sample || next_sample >= last_sample)
    return 0;

  uint64 to_recycle = last_sample - next_sample;

  if (verbose)
    cerr << "Input::recycle_data recycle " << to_recycle << " samples" << endl;

  if (to_recycle > block_size)
    to_recycle = block_size;

  uint64 recycle_bytes = data->nbytes (to_recycle);
  uint64 offset_bytes = data->nbytes (next_sample - start_sample);

  // next_sample += to_recycle;

  if (verbose)
    cerr << "Input::recycle_data recycle " << recycle_bytes
	 << " bytes (offset=" << offset_bytes << " bytes)" << endl;

  // check if the next sample is already the start sample
  if (!offset_bytes)
    return to_recycle;

  unsigned char *into = data->data;
  unsigned char *from = data->data + offset_bytes;

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
    cerr << "Input::recycle_data recycled " << to_recycle << endl;

  return to_recycle;
}


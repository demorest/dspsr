#include <iostream>

#include "dsp/Input.h"
#include "dsp/BitSeries.h"

#include "environ.h"

#include "Error.h"
#include "genutil.h"

bool dsp::Input::verbose = false;

dsp::Input::Input (const char* name) : Operation (name)
{
  block_size = overlap = 0;
  next_sample = 0;
}

dsp::Input::~Input ()
{
}

//! Load data into the BitSeries specified by set_output
void dsp::Input::operation ()
{
  if (verbose)
    cerr << "Input::operate" << endl;
  
  if (!output)
    throw Error (InvalidState, "Input::operate", "no output BitSeries");

  load (output);
}

//! Set the BitSeries to which data will be loaded
void dsp::Input::set_output (BitSeries* data)
{
  if (!output || output != data) {
    output = data;
    output -> input_sample = -1;
  }
}


/*! Set the Observation attributes of data and load the next block of data
 */
void dsp::Input::load (BitSeries* data)
{
  if (verbose)
    cerr << "Input::load (BitSeries* = " << data << ")" << endl;

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

  if (verbose)
    cerr << "Input::load call load_data" << endl;

  load_data (data);

  data->input_sample = next_sample;

  next_sample += block_size - overlap;

  if (verbose)
    cerr << "Input::load exit with next_sample="<< next_sample <<endl;
}

/*!
  Set from where the next "load_block" will load
  \param offset the number of time samples to offset
  \param whence from where to offset: SEEK_SET, SEEK_CUR, SEEK_END (see
  <unistd.h>)
*/
void dsp::Input::seek (int64 offset, int whence)
{
  if (verbose)
    cerr << "Input::seek offset=" << offset << endl;

  switch (whence) {

  case SEEK_SET:
    if (offset < 0)
      throw Error (InvalidRange, "Input::seek", "SEEK_SET negative offset");
    next_sample = offset;
    break;

  case SEEK_CUR:
    if (offset < -(int64)next_sample)
      throw Error (InvalidRange, "Input::seek", "SEEK_CUR negative offset");
    next_sample += offset;
    break;

  case SEEK_END:
    if (!info.get_ndat())
      throw Error (InvalidState, "Input::seek", "SEEK_END unknown eod");

    if (offset < -int64(info.get_ndat()))
      throw Error (InvalidRange, "Input::seek", "SEEK_END negative offset");

    next_sample = info.get_ndat() + offset;
    break;

  default:
    throw Error (InvalidParam, "Input::seek", "invalid whence");
  }

}

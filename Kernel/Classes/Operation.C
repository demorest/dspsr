#include "Operation.h"
#include "Timeseries.h"
#include "genutil.h"

//! Global flag tells all Operations to record the time spent operating
bool dsp::Operation::record_time = false;

//! Global verbosity flag
bool dsp::Operation::verbose = false;

//! All sub-classes must specify name and capacity for inplace operation
dsp::Operation::Operation (const char* _name, bool inplace)
{
  name = _name;
  inplace_capable = inplace;
}

void dsp::Operation::operate ()
{
  if (!input)
    throw_str ("Operation::operate no input");

  if (!output)
    throw_str ("Operation::operate no output");

  if (!inplace_capable && input == output)
    throw_str ("Operation::operate input cannot equal output");

  if (record_time)
    optime.start();

  //! call the pure virtual method defined by sub-classes
  operation ();

  if (record_time)
    optime.stop();
}

//! Set the container from which input data will be read
void dsp::Operation::set_input (const Timeseries* _input)
{
  input = _input;
}

//! Set the container into which output data will be written
void dsp::Operation::set_output (Timeseries* _output)
{
  output = _output;
}

//! Return pointer to the container from which input data will be read
const dsp::Timeseries* dsp::Operation::get_input () const
{
  return input;
}

//! Return pointer to the container into which output data will be written
dsp::Timeseries* dsp::Operation::get_output () const
{
  return output;
}

//! Return pointer to a memory resource shared by operations
void* dsp::Operation::workingspace (size_t nbytes)
{
  static char* working_space = NULL;
  static size_t working_size = 0;

  if (!nbytes) {
    if (working_space) delete [] working_space; working_space = 0;
    working_size = 0;
  }

  if (working_size < nbytes) {
    if (working_space) delete [] working_space; working_space = 0;
    working_space = new char [nbytes];

    if (!working_space)
      throw_str ("Operation::workingspace: error allocating %d bytes", nbytes);

    working_size = nbytes;
  }

  return working_space;
}


#include "Operation.h"
#include "Timeseries.h"
#include "genutil.h"

//! Global flag tells all Operations to record the time spent operating
bool dsp::Operation::record_time = false;

//! Global verbosity flag
bool dsp::Operation::verbose = false;

//! All sub-classes must specify name and capacity for inplace operation
dsp::Operation::Operation (const char* _name, Behaviour _type)
{
  name = _name;
  type = _type;
  IsPrepared = false;
}

dsp::Operation::~Operation ()
{
}

void dsp::Operation::prepare ()
{
  IsPrepared = true;
}

void dsp::Operation::operate ()
{
  if (verbose)
    cerr << "Operation::operate["+name+"]" << endl;

  if (!input)
    throw_str ("Operation::operate["+name+"] no input");

  if (input->get_ndat() < 1)
    throw_str ("Operation::operate["+name+"] empty input");

  string reason;
  if (!input->state_is_valid (reason))
    throw_str ("Operation::operate["+name+"] invalid input state: "+reason);

  if (!output)
    throw_str ("Operation::operate["+name+"] no output");


  if (record_time)
    optime.start();

  if( !IsPrepared )
    prepare();

  //! call the pure virtual method defined by sub-classes
  operation();

  if (record_time)
    optime.stop();

  if (!output->state_is_valid (reason))
    throw_str ("Operation::operate["+name+"] invalid output state: "+reason);

  /*! If an operation is performed on a Timeseries, then the data no
    longer represents that which was loaded by an Input sub-class.
    Reset the input_sample attribute of the output so that the
    Input::recycle_data method knows that the data has been modified
    since the last call to Input::load */
  output -> input_sample = -1;
}

//! Set the container from which input data will be read
void dsp::Operation::set_input (const Timeseries* _input)
{
  input = _input;

  if (type == inplace)
    output = const_cast<Timeseries*>(input.get());

  if (type == outofplace && input && output && input == output)
    throw_str ("Operation::set_input["+name+"] input must != output");

}

//! Set the container into which output data will be written
void dsp::Operation::set_output (Timeseries* _output)
{
  output = _output;
  if (type == inplace)
    input = output;

  if (type == outofplace && input && output && input == output)
    throw_str ("Operation::set_output["+name+"] output must != input");
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

double dsp::Operation::get_total_time () const
{
  return optime.get_total();
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
      throw_str ("Operation::workingspace: error allocating %d bytes",nbytes);

    working_size = nbytes;
  }

  return working_space;
}


#include "dsp/Unpacker.h"
#include "dsp/Observation.h"
#include "dsp/Timeseries.h"
#include "dsp/Chronoseries.h"
#include "dsp/Basicseries.h"

#include "Error.h"

//! Initialize and resize the output before calling unpack
void dsp::Unpacker::operation ()
{
  if (verbose)
    cerr << "Unpacker::operation" << endl;;

  // set the Observation information
  output->Observation::operator=(*input);

  // output will contain floating point values
  output->set_nbit (8 * sizeof(float));

  // resize the output 
  output->resize (input->get_ndat());

  // unpack the data
  unpack ();

  if (verbose)
    cerr << "Unpacker::operation exit" << endl;;
}

//! Specialize the unpacker to the Observation
void dsp::Unpacker::match (const Observation* observation)
{
  if (verbose)
    cerr << "Unpacker::match" << endl;
}

void dsp::Unpacker::check_input(){
  Chronoseries* _input = dynamic_cast<Chronoseries*>(const_cast<Basicseries*>(Operation::input.get()));
  if( _input==0 )
    throw Error(InvalidParam,"dsp::Unpacker::check_input()",
		"You have given an input to this operation that is required to be a dsp::Chronoseries- it is some other type of dsp::Basicseries");
  input = _input;
}

void dsp::Unpacker::check_output(){
  Timeseries* _output = dynamic_cast<Timeseries*>(Operation::output.get());
  if( _output==0 )
    throw Error(InvalidParam,"dsp::Unpacker::check_output()",
		"You have given an output to this operation that is required to be a dsp::Timeseries- it is some other type of dsp::Basicseries");
  output = _output;
}


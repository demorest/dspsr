#include "dsp/Unpacker.h"
#include "dsp/Observation.h"
#include "dsp/Timeseries.h"
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
    cerr << "Unpacker::match" << endl;;
}

#include "dsp/Unpacker.h"

#include "Error.h"

//! Initialize and resize the output before calling unpack
void dsp::Unpacker::transformation ()
{
  if (verbose)
    cerr << "Unpacker::transformation" << endl;;

  // set the Observation information
  output->Observation::operator=(*input);

  // output will contain floating point values
  output->set_nbit (8 * sizeof(float));

  // resize the output 
  output->resize (input->get_ndat());

  // unpack the data
  unpack ();

  if (verbose)
    cerr << "Unpacker::transformation exit" << endl;;
}

//! Specialize the unpacker to the Observation
void dsp::Unpacker::match (const Observation* observation)
{
  if (verbose)
    cerr << "Unpacker::match" << endl;
}


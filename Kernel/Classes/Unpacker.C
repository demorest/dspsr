#include "dsp/Unpacker.h"

#include "Error.h"

//! Initialize and resize the output before calling unpack
void dsp::Unpacker::transformation ()
{
  if (verbose)
    cerr << "Unpacker::transformation" << endl;;

  // set the Observation information
  output->Observation::operator=(*input);

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


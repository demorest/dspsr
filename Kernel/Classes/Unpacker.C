#include <stdio.h>
#include <stdlib.h>

#include "dsp/Unpacker.h"
#include "Error.h"

//! Initialize and resize the output before calling unpack
void dsp::Unpacker::transformation ()
{
  if (verbose)
    cerr << "dsp::Unpacker::transformation" << endl;;

  // set the Observation information
  output->Observation::operator=(*input);

  // resize the output 
  output->resize (input->get_ndat());

  // unpack the data
  unpack ();

  // The following lines deal with time sample resolution of the data source
  output->seek (input->get_request_offset());

  output->set_ndat (input->get_request_ndat());

  if (verbose)
    cerr << "dsp::Unpacker::transformation exit" << endl;;
}

//! Specialize the unpacker to the Observation
void dsp::Unpacker::match (const Observation* observation)
{
  if (verbose)
    cerr << "dsp::Unpacker::match" << endl;
}

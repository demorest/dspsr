#include "dsp/Unpacker.h"
#include "Error.h"

//! Return a pointer to a new instance of the appropriate sub-class
dsp::Unpacker* dsp::Unpacker::create (const Observation* observation)
{
  try {

    if (verbose) cerr << "dsp::Unpacker::create with " << registry.size()
                      << " registered sub-classes" << endl;

    for (unsigned ichild=0; ichild < registry.size(); ichild++)
      if ( registry[ichild]->matches (observation) ) {

        Unpacker* child = registry.create (ichild);
        child-> match( observation );

	if (verbose)
	  cerr << "dsp::Unpacker::create return new sub-class" << endl;
        return child;

      }
  } catch (Error& error) {
    throw error += "dsp::Unpacker::create";
  }

  throw Error (InvalidState, "dsp::Unpacker::create",
               "no unpacker for machine=" + observation->get_machine());
}

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


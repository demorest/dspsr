#include "Unpacker.h"
#include "Observation.h"
#include "Timeseries.h"
#include "Error.h"

Registry::List<dsp::Unpacker> dsp::Unpacker::registry;

//! Return a pointer to a new instance of the appropriate sub-class
dsp::Unpacker* dsp::Unpacker::create (const Observation* observation)
{ 
  try {

    if (verbose) cerr << "Unpacker::create with " << registry.size() 
		      << " registered sub-classes" << endl;

    for (unsigned ichild=0; ichild < registry.size(); ichild++)
      if ( registry[ichild]->matches (observation) ) {

	Unpacker* child = registry.create (ichild);
	child-> match( observation );

	return child;

      }
  } catch (Error& error) {
    throw error += "Unpacker::create";
  }

  throw Error (InvalidState, "Unpacker::create",
	       "no unpacker for machine=" + observation->get_machine());
}

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

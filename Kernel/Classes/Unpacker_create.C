/***************************************************************************
 *
 *   Copyright (C) 2003 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/Unpacker.h"

using namespace std;

//! Return a pointer to a new instance of the appropriate sub-class
dsp::Unpacker* dsp::Unpacker::create (const Observation* observation)
{
  try {

    if (verbose) cerr << "dsp::Unpacker::create with " << registry.size()
                      << " registered sub-classes" << endl;

    for (unsigned ichild=0; ichild < registry.size(); ichild++){
      if( verbose )
	fprintf(stderr,"Testing child %d: %s\n",ichild,
		registry[ichild]->get_name().c_str());
      if ( registry[ichild]->matches (observation) ) {

        Unpacker* child = registry.create (ichild);
        child-> match( observation );

        if (verbose)
          cerr << "dsp::Unpacker::create return new sub-class" << endl;
        return child;

      }
    }

  } catch (Error& error) {
    throw error += "dsp::Unpacker::create";
  }

  throw Error (InvalidState, "dsp::Unpacker::create",
               "no unpacker for machine=" + observation->get_machine());
}


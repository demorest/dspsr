/***************************************************************************
 *
 *   Copyright (C) 2003 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/Unpacker.h"

using namespace std;

//! Return a pointer to a new instance of the appropriate sub-class
dsp::Unpacker* dsp::Unpacker::create (const Observation* observation) try
{
  Register& registry = get_register();

  if (verbose)
    std::cerr << "dsp::Unpacker::create with " << registry.size()
	      << " registered sub-classes" << std::endl;

  for (unsigned ichild=0; ichild < registry.size(); ichild++)
  {
    if (verbose)
      std::cerr << "dsp::Unpacker::create testing "
                << registry[ichild]->get_name() << std::endl;

    if ( registry[ichild]->matches (observation) )
    {
      Unpacker* child = registry.create (ichild);
      child-> match( observation );

      if (verbose)
        std::cerr << "dsp::Unpacker::create return new sub-class" 
		  << std::endl;

      return child;

    }
  }

  throw Error (InvalidState, string(),
               "no unpacker for machine=" + observation->get_machine());
}
catch (Error& error)
{
  throw error += "dsp::Unpacker::create";
}


#include "dsp/CPSR2TwoBitCorrection.h"
#include "dsp/CPSRTwoBitCorrection.h"
#include "dsp/S2TwoBitCorrection.h"
#include "dsp/OneBitCorrection.h"
#include "dsp/CoherentFBUnpacker.h"
#include "dsp/NullUnpacker.h"

#ifdef Digi_returned_to_Makefile
#include "dsp/DigiUnpack.h"
#endif

#include "Error.h"

Registry::List<dsp::Unpacker> dsp::Unpacker::registry;

// HSK 9/12/02 Please note that coherentfb comes first as basically if data has been rewritten to disk it should be unpacked by that packing, rather than whatever unpacker it originally got unpacked as.

static Registry::List<dsp::Unpacker>::Enter<dsp::CoherentFBUnpacker>    coherentfb;
static Registry::List<dsp::Unpacker>::Enter<dsp::NullUnpacker>          bitseries_unpacker;
static Registry::List<dsp::Unpacker>::Enter<dsp::CPSR2TwoBitCorrection> cpsr2;
static Registry::List<dsp::Unpacker>::Enter<dsp::CPSRTwoBitCorrection>  cpsr;
static Registry::List<dsp::Unpacker>::Enter<dsp::S2TwoBitCorrection>    s2;
static Registry::List<dsp::Unpacker>::Enter<dsp::OneBitCorrection>      pmdaq;

//static Registry::List<dsp::Unpacker>::Enter<dsp::DigiUnpack> digiunpack;

//! Return a pointer to a new instance of the appropriate sub-class
dsp::Unpacker* dsp::Unpacker::create (const Observation* observation)
{
  if(verbose)
    fprintf(stderr,"\nIn dsp::Unpacker::create ()\n");

  try {

    if (verbose) cerr << "Unpacker::create with " << registry.size()
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
    throw error += "Unpacker::create";
  }

  throw Error (InvalidState, "Unpacker::create",
               "no unpacker for machine=" + observation->get_machine());
}



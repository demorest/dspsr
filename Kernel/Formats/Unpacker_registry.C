#include "dsp/CPSR2TwoBitCorrection.h"
#include "dsp/CPSRTwoBitCorrection.h"
#include "dsp/S2TwoBitCorrection.h"
#include "dsp/OneBitCorrection.h"
#include "dsp/DigiUnpack.h"

#include "Error.h"

Registry::List<dsp::Unpacker> dsp::Unpacker::registry;

static Registry::List<dsp::Unpacker>::Enter<dsp::CPSR2TwoBitCorrection> cpsr2;
static Registry::List<dsp::Unpacker>::Enter<dsp::CPSRTwoBitCorrection>  cpsr;
static Registry::List<dsp::Unpacker>::Enter<dsp::S2TwoBitCorrection>    s2;
static Registry::List<dsp::Unpacker>::Enter<dsp::OneBitCorrection>      pmdaq;

//static Registry::List<dsp::Unpacker>::Enter<dsp::DigiUnpack> digiunpack;

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



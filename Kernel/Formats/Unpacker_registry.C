
/* HSK 9/12/02 Please note that coherentfb comes first as basically if
   data has been rewritten to disk it should be unpacked by that packing,
   rather than whatever unpacker it originally got unpacked as.
*/

#include "dsp/CoherentFBUnpacker.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::CoherentFBUnpacker> cfb;
#include "dsp/NullUnpacker.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::NullUnpacker> bitseries;

#include "backends.h"

#if DSP_CPSR2
#include "dsp/CPSR2TwoBitCorrection.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::CPSR2TwoBitCorrection> cpsr2;
#endif

#if DSP_CPSR
#include "dsp/CPSRTwoBitCorrection.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::CPSRTwoBitCorrection>  cpsr;
#endif

#if DSP_S2
#include "dsp/S2TwoBitCorrection.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::S2TwoBitCorrection>    s2;
#endif

#if DSP_PMDAQ
#include "dsp/OneBitCorrection.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::OneBitCorrection>      pmdaq;
#endif

#include "Error.h"

Registry::List<dsp::Unpacker> dsp::Unpacker::registry;


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



/*! \file Unpacker_registry.C
  \brief Register dsp::Unpacker-derived classes for use in this file
    
  Classes that inherit dsp::Unpacker may be registered for use by
  utilizing the Registry::List<dsp::Unpacker>::Enter<Type> template class.
  Static instances of this template class should be given a unique
  name and enclosed within preprocessor directives that make the
  instantiation optional.  There are plenty of examples in the source code.

  \note HSK 9/12/02 CoherentFBUnpacker comes first as basically if
  data has been rewritten to disk it should be unpacked by that packing,
  rather than whatever unpacker it originally got unpacked as.
*/

#include "dsp/Unpacker.h"

/*! The registry must always be constructed before the entries. */
Registry::List<dsp::Unpacker> dsp::Unpacker::registry;

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

#if DSP_PuMa
#include "dsp/PuMaTwoBitCorrection.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::PuMaTwoBitCorrection>  puma;
#endif


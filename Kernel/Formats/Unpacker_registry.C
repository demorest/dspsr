/*! \file Unpacker_registry.C
  \brief Register dsp::Unpacker-derived classes for use in this file
    
  Classes that inherit dsp::Unpacker may be registered for use by
  utilizing the Registry::List<dsp::Unpacker>::Enter<Type> template class.
  Static instances of this template class should be given a unique
  name and enclosed within preprocessor directives that make the
  instantiation optional.  There are plenty of examples in the source code.
*/

#include "dsp/Unpacker.h"

/*! The registry must always be constructed before the entries. */
Registry::List<dsp::Unpacker> dsp::Unpacker::registry;

#include "dsp/NullUnpacker.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::NullUnpacker> bitseries;

#if DSP_MINI
#include "dsp/MiniUnpack.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::MiniUnpack> register_miniunpack;
#endif

#include "dsp/backends.h"

#if DSP_CPSR2_4bit
#include "dsp/CPSR2FourBitUnpacker.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::CPSR2FourBitUnpacker> cpsr2_4bit;
#endif

#if DSP_CPSR2
#include "dsp/CPSR2TwoBitCorrection.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::CPSR2TwoBitCorrection> cpsr2;
#endif

#if DSP_CPSR
#include "dsp/CPSRTwoBitCorrection.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::CPSRTwoBitCorrection>  cpsr;
#endif

#if DSP_CPSR2_8bit
#include "dsp/CPSR2_8bitUnpacker.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::CPSR2_8bitUnpacker> cpsr2_8bit;
#endif

#if DSP_Maxim
#include "dsp/MaximUnpacker.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::MaximUnpacker> maxim;
#endif

#if DSP_SMRO
#include "dsp/SMROTwoBitCorrection.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::SMROTwoBitCorrection>  smro;
#endif

#if DSP_VSIB
#include "dsp/VSIBTwoBitCorrection.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::VSIBTwoBitCorrection>  vsib;
#endif

#if DSP_K5
#include "dsp/K5TwoBitCorrection.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::K5TwoBitCorrection>  k5;
#endif

#if DSP_BCPM
#include "dsp/BCPMUnpacker.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::BCPMUnpacker> registry_bcpm;
#endif

#if DSP_Spigot
#include "dsp/ACFUnpack.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::ACFUnpack> spigot;
#endif

#if DSP_MARK4
#include "dsp/Mark4TwoBitCorrection.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::Mark4TwoBitCorrection> mark4;
#endif

#if DSP_S2
#include "dsp/S2TwoBitCorrection.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::S2TwoBitCorrection>  s2;
#endif

#if DSP_PMDAQ
#include "dsp/OneBitCorrection.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::OneBitCorrection>  pmdaq;
#endif

#if DSP_PuMa
#include "dsp/PuMaTwoBitCorrection.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::PuMaTwoBitCorrection>  puma;
#endif


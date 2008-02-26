/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
/*! \file Unpacker_registry.C
  \brief Register dsp::Unpacker-derived classes for use in this file
    
  Classes that inherit dsp::Unpacker may be registered for use by
  utilizing the Registry::List<dsp::Unpacker>::Enter<Type> template class.
  Static instances of this template class should be given a unique
  name and enclosed within preprocessor directives that make the
  instantiation optional.  There are plenty of examples in the source code.
*/

#include "dsp/Unpacker.h"

//! Based on backends.list, backends.h #defines the DSP_Backend macros
#include "backends.h"

/*! The registry must always be constructed before the entries. */
Registry::List<dsp::Unpacker> dsp::Unpacker::registry;

#if DSP_apsr
#include "dsp/APSRTwoBitCorrection.h"
#include "dsp/APSRFourBit.h"
#include "dsp/APSREightBit.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::APSRTwoBitCorrection> apsr2;
static Registry::List<dsp::Unpacker>::Enter<dsp::APSRFourBit> apsr4;
static Registry::List<dsp::Unpacker>::Enter<dsp::APSREightBit> apsr8;
#endif

#if DSP_asp
#include "dsp/ASPUnpacker.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::ASPUnpacker> asp;
#endif

#if DSP_bcpm
#include "dsp/BCPMUnpacker.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::BCPMUnpacker> registry_bcpm;
#endif

#if DSP_cpsr
#include "dsp/CPSRTwoBitCorrection.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::CPSRTwoBitCorrection> cpsr;
#endif

#if DSP_cpsr2
#include "dsp/CPSR2TwoBitCorrection.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::CPSR2TwoBitCorrection> cpsr2;
#endif

#if DSP_fadc
#include "dsp/FadcUnpacker.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::FadcUnpacker> fadc;
#include "dsp/FadcTwoBitCorrection.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::FadcTwoBitCorrection> fadc2;
#endif

#if DSP_lbadr
#include "dsp/SMROTwoBitCorrection.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::SMROTwoBitCorrection> lbadr;
#endif

#if DSP_lbadr64
#include "dsp/LBADR64_TwoBitCorrection.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::LBADR64_TwoBitCorrection> lbadr64;
#endif

#if DSP_mark4
#include "dsp/Mark4TwoBitCorrection.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::Mark4TwoBitCorrection> mark4;
#endif

#if DSP_mark5
#include "dsp/Mark5Unpacker.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::Mark5Unpacker> mark5_general;
#include "dsp/Mark5TwoBitCorrection.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::Mark5TwoBitCorrection> mark5;
#endif

#if DSP_maxim
#include "dsp/MaximUnpacker.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::MaximUnpacker> maxim;
#endif

#if DSP_mini
#include "dsp/MiniUnpack.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::MiniUnpack> miniunpack;
#endif

#if DSP_mwa
// There is no MWA unpacker checked into the repository
#endif

#if DSP_pmdaq
#include "dsp/OneBitCorrection.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::OneBitCorrection>  pmdaq;
#endif

#if DSP_puma
#include "dsp/PuMaTwoBitCorrection.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::PuMaTwoBitCorrection>  puma;
#endif

#if DSP_puma2
#include "dsp/PuMa2Unpacker.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::PuMa2Unpacker> puma2;
#endif

#if DSP_s2
#include "dsp/S2TwoBitCorrection.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::S2TwoBitCorrection>  s2;
#endif

#if DSP_spigot
#include "dsp/ACFUnpack.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::ACFUnpack> spigot;
#endif

#if DSP_wapp
#include "dsp/WAPPUnpacker.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::WAPPUnpacker> wapp;
#endif

// ///////////////////////////////////////////////////////////////////////////
//
// The rest of these need work or seem to have disappeared
//
// ///////////////////////////////////////////////////////////////////////////

#if DSP_CPSR2_4bit
#include "dsp/CPSR2FourBitUnpacker.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::CPSR2FourBitUnpacker> cpsr2_4;
#endif

#if DSP_CPSR2_8bit
#include "dsp/CPSR2_8bitUnpacker.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::CPSR2_8bitUnpacker> cpsr2_8;
#endif

#if DSP_vsib
#include "dsp/VSIBTwoBitCorrection.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::VSIBTwoBitCorrection>  vsib;
#endif

#if DSP_DUMBLBA
#include "dsp/DumbLBAUnpacker.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::Dumb_LBAUnpacker> unpacker_register_dumblba;
#endif

#if DSP_k5
#include "dsp/K5TwoBitCorrection.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::K5TwoBitCorrection>  k5;
#endif

#if 0
/* Wvs FIX LATER */
#include "dsp/NullUnpacker.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::NullUnpacker> bitseries;
#endif

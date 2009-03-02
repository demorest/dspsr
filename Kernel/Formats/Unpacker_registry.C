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

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/Unpacker.h"

/*! The registry must always be constructed before the entries. */
Registry::List<dsp::Unpacker> dsp::Unpacker::registry;

#if HAVE_apsr

#include "dsp/APSRTwoBitCorrection.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::APSRTwoBitCorrection> apsr2;

#include "dsp/APSRFourBit.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::APSRFourBit> apsr4;

#include "dsp/APSREightBit.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::APSREightBit> apsr8;

#endif

#if HAVE_asp
#include "dsp/ASPUnpacker.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::ASPUnpacker> asp;
#endif

#if HAVE_bcpm
#include "dsp/BCPMUnpacker.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::BCPMUnpacker> registry_bcpm;
#endif

#if HAVE_bpsr
#include "dsp/BPSRUnpacker.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::BPSRUnpacker> bpsr;
#endif

#if HAVE_cpsr
#include "dsp/CPSRTwoBitCorrection.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::CPSRTwoBitCorrection> cpsr;
#endif

#if HAVE_cpsr2
#include "dsp/CPSR2TwoBitCorrection.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::CPSR2TwoBitCorrection> cpsr2;
#endif

#if HAVE_fadc
#include "dsp/FadcUnpacker.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::FadcUnpacker> fadc;
#include "dsp/FadcTwoBitCorrection.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::FadcTwoBitCorrection> fadc2;
#endif

#if HAVE_gmrt
#include "dsp/GMRTUnpacker.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::GMRTUnpacker> gmrt;
#if HAVE_PRESTO
#include "dsp/GMRTFilterbank16.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::GMRTFilterbank16> gmrt16;
#endif
#endif

#if HAVE_lbadr
#include "dsp/SMROTwoBitCorrection.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::SMROTwoBitCorrection> lbadr;
#endif

#if HAVE_lbadr64
#include "dsp/LBADR64_TwoBitCorrection.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::LBADR64_TwoBitCorrection> lbadr64;
#endif

#if HAVE_mark4
#include "dsp/Mark4TwoBitCorrection.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::Mark4TwoBitCorrection> mark4;
#endif

#if HAVE_mark5
#include "dsp/Mark5Unpacker.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::Mark5Unpacker> mark5_general;
#include "dsp/Mark5TwoBitCorrection.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::Mark5TwoBitCorrection> mark5;
#endif

#if HAVE_maxim
#include "dsp/MaximUnpacker.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::MaximUnpacker> maxim;
#endif

#if HAVE_mini
#include "dsp/MiniUnpack.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::MiniUnpack> miniunpack;
#endif

#if HAVE_mwa
// There is no MWA unpacker checked into the repository
#endif

#if HAVE_pmdaq
#include "dsp/OneBitCorrection.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::OneBitCorrection>  pmdaq;
#endif

#if HAVE_puma
#include "dsp/PuMaTwoBitCorrection.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::PuMaTwoBitCorrection>  puma;
#endif

#if HAVE_puma2
#include "dsp/PuMa2Unpacker.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::PuMa2Unpacker> puma2;
#endif

#if HAVE_s2
#include "dsp/S2TwoBitCorrection.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::S2TwoBitCorrection>  s2;
#endif

#if HAVE_sigproc
#include "dsp/SigProcEight.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::SigProcEight> sigproc8;
#include "dsp/SigProcTwo.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::SigProcTwo> sigproc2;
#endif

#if HAVE_spigot
#include "dsp/ACFUnpack.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::ACFUnpack> spigot;
#endif

#if HAVE_wapp
#include "dsp/WAPPUnpacker.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::WAPPUnpacker> wapp;
#endif

// ///////////////////////////////////////////////////////////////////////////
//
// The rest of these need work or seem to have disappeared
//
// ///////////////////////////////////////////////////////////////////////////

#if HAVE_CPSR2_4bit
#include "dsp/CPSR2FourBitUnpacker.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::CPSR2FourBitUnpacker> cpsr2_4;
#endif

#if HAVE_CPSR2_8bit
#include "dsp/CPSR2_8bitUnpacker.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::CPSR2_8bitUnpacker> cpsr2_8;
#endif

#if HAVE_vsib
#include "dsp/VSIBTwoBitCorrection.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::VSIBTwoBitCorrection>  vsib;
#endif

#if HAVE_DUMBLBA
#include "dsp/DumbLBAUnpacker.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::Dumb_LBAUnpacker> unpacker_register_dumblba;
#endif

#if HAVE_k5
#include "dsp/K5TwoBitCorrection.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::K5TwoBitCorrection>  k5;
#endif


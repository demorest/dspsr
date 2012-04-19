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


/*! built-in FloatUnpacker reads the format output by Dump operation */
#include "dsp/FloatUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::FloatUnpacker> dump;


#if HAVE_apsr

#include "dsp/APSRTwoBitCorrection.h"
static dsp::Unpacker::Register::Enter<dsp::APSRTwoBitCorrection> apsr2;

#include "dsp/APSRFourBit.h"
static dsp::Unpacker::Register::Enter<dsp::APSRFourBit> apsr4;

#include "dsp/APSREightBit.h"
static dsp::Unpacker::Register::Enter<dsp::APSREightBit> apsr8;

#endif

#if HAVE_asp
#include "dsp/ASPUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::ASPUnpacker> asp;
#endif

#if HAVE_bcpm
#include "dsp/BCPMUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::BCPMUnpacker> registry_bcpm;
#endif

#if HAVE_bpsr
#include "dsp/BPSRUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::BPSRUnpacker> bpsr;
#endif

#if HAVE_caspsr
#include "dsp/CASPSRUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::CASPSRUnpacker> caspsr;
#endif

#if HAVE_cpsr
#include "dsp/CPSRTwoBitCorrection.h"
static dsp::Unpacker::Register::Enter<dsp::CPSRTwoBitCorrection> cpsr;
#endif

#if HAVE_cpsr2
#include "dsp/CPSR2TwoBitCorrection.h"
static dsp::Unpacker::Register::Enter<dsp::CPSR2TwoBitCorrection> cpsr2;
#endif

#if HAVE_dummy
#include "dsp/DummyUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::DummyUnpacker> dummy;
#endif

#if HAVE_fadc
#include "dsp/FadcUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::FadcUnpacker> fadc;
#include "dsp/FadcTwoBitCorrection.h"
static dsp::Unpacker::Register::Enter<dsp::FadcTwoBitCorrection> fadc2;
#endif

#if HAVE_gmrt
#include "dsp/GMRTUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::GMRTUnpacker> gmrt;
#include "dsp/GMRTFourBit.h"
static dsp::Unpacker::Register::Enter<dsp::GMRTFourBit> gmrt4;
#include "dsp/GMRTFilterbank16.h"
static dsp::Unpacker::Register::Enter<dsp::GMRTFilterbank16> gmrt16;
#endif

#if HAVE_guppi
#include "dsp/GUPPIUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::GUPPIUnpacker> guppi;
#endif

#if HAVE_lofar_dal
#include "dsp/LOFAR_DALUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::LOFAR_DALUnpacker> lofar_dal;
#endif

#if HAVE_lbadr
#include "dsp/SMROTwoBitCorrection.h"
static dsp::Unpacker::Register::Enter<dsp::SMROTwoBitCorrection> lbadr;
#endif

#if HAVE_lbadr64
#include "dsp/LBADR64_TwoBitCorrection.h"
static dsp::Unpacker::Register::Enter<dsp::LBADR64_TwoBitCorrection> lbadr64;
#endif

#if HAVE_spda1k
#include "dsp/spda1k_Unpacker.h"
static dsp::Unpacker::Register::Enter<dsp::SPDA1K_Unpacker> spda1k;
#endif

#if HAVE_mark4
#include "dsp/Mark4TwoBitCorrection.h"
static dsp::Unpacker::Register::Enter<dsp::Mark4TwoBitCorrection> mark4;
#endif

#if HAVE_mark5
#include "dsp/Mark5Unpacker.h"
static dsp::Unpacker::Register::Enter<dsp::Mark5Unpacker> mark5_general;
#include "dsp/Mark5TwoBitCorrection.h"
static dsp::Unpacker::Register::Enter<dsp::Mark5TwoBitCorrection> mark5;
#endif

#if HAVE_maxim
#include "dsp/MaximUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::MaximUnpacker> maxim;
#endif

#if HAVE_mini
#include "dsp/MiniUnpack.h"
static dsp::Unpacker::Register::Enter<dsp::MiniUnpack> miniunpack;
#endif

#if HAVE_mwa
// There is no MWA unpacker checked into the repository
#endif

#if HAVE_pmdaq
#include "dsp/OneBitCorrection.h"
static dsp::Unpacker::Register::Enter<dsp::OneBitCorrection>  pmdaq;
#endif

#if HAVE_puma
#include "dsp/PuMaTwoBitCorrection.h"
static dsp::Unpacker::Register::Enter<dsp::PuMaTwoBitCorrection>  puma;
#endif

#if HAVE_puma2
#include "dsp/PuMa2Unpacker.h"
static dsp::Unpacker::Register::Enter<dsp::PuMa2Unpacker> puma2;
#endif

#if HAVE_s2
#include "dsp/S2TwoBitCorrection.h"
static dsp::Unpacker::Register::Enter<dsp::S2TwoBitCorrection>  s2;
#endif

#if HAVE_sigproc
#include "dsp/SigProcUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::SigProcUnpacker> sigproc;
#endif

#if HAVE_spigot
#include "dsp/ACFUnpack.h"
static dsp::Unpacker::Register::Enter<dsp::ACFUnpack> spigot;
#endif

#if HAVE_fits
#include "dsp/FITSUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::FITSUnpacker> fits;
#endif

#if HAVE_vdif
#include "dsp/VDIFTwoBitCorrection.h"
static dsp::Unpacker::Register::Enter<dsp::VDIFTwoBitCorrection> vdif;
#include "dsp/VDIFTwoBitCorrectionMulti.h"
static dsp::Unpacker::Register::Enter<dsp::VDIFTwoBitCorrectionMulti> vdif_multi;
#endif

#if HAVE_wapp
#include "dsp/WAPPUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::WAPPUnpacker> wapp;
#endif

// ///////////////////////////////////////////////////////////////////////////
//
// The rest of these need work or seem to have disappeared
//
// ///////////////////////////////////////////////////////////////////////////

#if HAVE_CPSR2_4bit
#include "dsp/CPSR2FourBitUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::CPSR2FourBitUnpacker> cpsr2_4;
#endif

#if HAVE_CPSR2_8bit
#include "dsp/CPSR2_8bitUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::CPSR2_8bitUnpacker> cpsr2_8;
#endif

#if HAVE_vsib
#include "dsp/VSIBTwoBitCorrection.h"
static dsp::Unpacker::Register::Enter<dsp::VSIBTwoBitCorrection>  vsib;
#endif

#if HAVE_DUMBLBA
#include "dsp/DumbLBAUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::Dumb_LBAUnpacker> unpacker_register_dumblba;
#endif

#if HAVE_k5
#include "dsp/K5TwoBitCorrection.h"
static dsp::Unpacker::Register::Enter<dsp::K5TwoBitCorrection>  k5;
#endif

/*
  Generic eight-bit unpacker is used if no other eight-bit unpacker steps up
*/

#include "dsp/GenericEightBitUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::GenericEightBitUnpacker> gen8bit;

/*
  get_registry is defined here to ensure that this file is linked
*/
dsp::Unpacker::Register& dsp::Unpacker::get_register()
{
  return Register::get_registry();
}

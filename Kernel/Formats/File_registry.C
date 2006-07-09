/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
/*! \file File_registry.C
  \brief Register dsp::File-derived classes for use in this file
    
  Classes that inherit dsp::File may be registered for use by
  utilizing the Registry::List<dsp::File>::Enter<Type> template class.
  Static instances of this template class should be given a unique
  name and enclosed within preprocessor directives that make the
  instantiation optional.  There are plenty of examples in the source code.

  \note Do not change the order in which registry entries are made
  without testing all of the file types.  The S2 should remain last as
  it doesn't perform a proper is_valid() test.  Ensure that anything
  added does perform a proper is_valid() test.
*/


#include "dsp/File.h"

//! Based on backends.list, backends.h #defines the DSP_Backend macros
#include "backends.h"
#include "Error.h"

/*! The registry must always be constructed before the entries. */
Registry::List<dsp::File> dsp::File::registry;

#if DSP_cpsr2
#include "dsp/CPSR2File.h"
static Registry::List<dsp::File>::Enter<dsp::CPSR2File> register_cpsr2;
#endif

//Tej's new CPSR2 8 bit File Format:
#if DSP_CPSR2_8bit
#include "dsp/EightBitFile.h"
static Registry::List<dsp::File>::Enter<dsp::EightBitFile> register_eightbitcpsr2;
#endif

// This is defined in libdsp.a
// It comes before CPSRFile as PSPMverify seems to think MultiBitSeriesFiles are CPSR files
#include "dsp/MultiBitSeriesFile.h"
static Registry::List<dsp::File>::Enter<dsp::MultiBitSeriesFile> register_multibitseriesfile;

#if DSP_mini
#include "dsp/MiniFile.h"
static Registry::List<dsp::File>::Enter<dsp::MiniFile> register_minifile;
#endif

#if DSP_cpsr
#include "dsp/CPSRFile.h"
static Registry::List<dsp::File>::Enter<dsp::CPSRFile> register_cpsr;
#endif

#if DSP_maxim
#include "dsp/MaximFile.h"
static Registry::List<dsp::File>::Enter<dsp::MaximFile> register_maxim;
#endif

#if DSP_smro
#include "dsp/SMROFile.h"
static Registry::List<dsp::File>::Enter<dsp::SMROFile> register_smro;
#endif

#if DSP_vsib
#include "dsp/VSIBFile.h"
static Registry::List<dsp::File>::Enter<dsp::VSIBFile> register_vsib;
#endif

#if DSP_pmdaq
#include "dsp/PMDAQFile.h"
static Registry::List<dsp::File>::Enter<dsp::PMDAQFile> register_pmdaq;
#endif

#if DSP_DUMBLBA
#include "dsp/DumbLBAFile.h"
static Registry::List<dsp::File>::Enter<dsp::Dumb_LBAFile> file_register_dumblba;
#endif

#if DSP_mwa
#include "dsp/MWAFile.h"
static Registry::List<dsp::File>::Enter<dsp::MWAFile> file_register_mwa;
#endif

#if DSP_puma
#include "dsp/PuMaFile.h"
static Registry::List<dsp::File>::Enter<dsp::PuMaFile> register_puma;
#endif

#if DSP_puma2
#include "dsp/PuMa2File.h"
static Registry::List<dsp::File>::Enter<dsp::PuMa2File> register_puma2;
#endif

#if DSP_spigot
#include "dsp/SpigotFile.h"
static Registry::List<dsp::File>::Enter<dsp::SpigotFile> register_spigot;
#endif

#if DSP_k5
#include "dsp/K5File.h"
static Registry::List<dsp::File>::Enter<dsp::K5File> register_k5;
#endif

#if DSP_bcpm
#include "dsp/BCPMFile.h"
static Registry::List<dsp::File>::Enter<dsp::BCPMFile> register_bcpm;
#endif

#include "dsp/BitSeriesFile.h"
static Registry::List<dsp::File>::Enter<dsp::BitSeriesFile> register_bitseriesfile;

#if DSP_mark4
#include "dsp/Mark4File.h"
static Registry::List<dsp::File>::Enter<dsp::Mark4File> register_mark4;
#endif

#if DSP_mark5
#include "dsp/Mark5File.h"
static Registry::List<dsp::File>::Enter<dsp::Mark5File> register_mark5;
#endif

#if DSP_asp
#include "dsp/ASPFile.h"
static Registry::List<dsp::File>::Enter<dsp::ASPFile> register_asp;
#endif

#if DSP_fadc
#include "dsp/FadcFile.h"
static Registry::List<dsp::File>::Enter<dsp::FadcFile> register_fadc;
#endif

#if DSP_s2
#include "dsp/S2File.h"
static Registry::List<dsp::File>::Enter<dsp::S2File> register_s2;
#endif


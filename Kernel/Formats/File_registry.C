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

//! Based on Makefile.local, backends.h #defines the DSP_Backend macros
#include "backends.h"
#include "Error.h"

/*! The registry must always be constructed before the entries. */
Registry::List<dsp::File> dsp::File::registry;

#if DSP_CPSR2
#include "dsp/CPSR2File.h"
static Registry::List<dsp::File>::Enter<dsp::CPSR2File> register_cpsr2;
#endif

#if DSP_CPSR
#include "dsp/CPSRFile.h"
static Registry::List<dsp::File>::Enter<dsp::CPSRFile> register_cpsr;
#endif

#if DSP_VSIB
#include "dsp/VSIBFile.h"
static Registry::List<dsp::File>::Enter<dsp::VSIBFile> register_vsib;
#endif

#if DSP_PMDAQ
#include "dsp/PMDAQFile.h"
static Registry::List<dsp::File>::Enter<dsp::PMDAQFile> register_pmdaq;
#endif

#if DSP_PuMa
#include "dsp/PuMaFile.h"
static Registry::List<dsp::File>::Enter<dsp::PuMaFile> register_puma;
#endif

// these are defined in libdsp.a
#include "dsp/CoherentFBFile.h"
static Registry::List<dsp::File>::Enter<dsp::CoherentFBFile> coherentfb;
#include "dsp/BitSeriesFile.h"
static Registry::List<dsp::File>::Enter<dsp::BitSeriesFile>  bitseries_file;

#if DSP_S2
#include "dsp/S2File.h"
static Registry::List<dsp::File>::Enter<dsp::S2File> register_s2;
#endif


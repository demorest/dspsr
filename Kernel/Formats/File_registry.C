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

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

/*! DummyFile is built in */
#include "dsp/DummyFile.h"
static dsp::File::Register::Enter<dsp::DummyFile> dummy_file;

/*! DADAFile is built in */
#include "dsp/DADAFile.h"
static dsp::File::Register::Enter<dsp::DADAFile> dada_file;

#if HAVE_asp
#include "dsp/ASPFile.h"
static dsp::File::Register::Enter<dsp::ASPFile> register_asp;
#endif

#if HAVE_bcpm
#include "dsp/BCPMFile.h"
static dsp::File::Register::Enter<dsp::BCPMFile> register_bcpm;
#endif

#if HAVE_cpsr
#include "dsp/CPSRFile.h"
static dsp::File::Register::Enter<dsp::CPSRFile> register_cpsr;
#endif

#if HAVE_cpsr2
#include "dsp/CPSR2File.h"
static dsp::File::Register::Enter<dsp::CPSR2File> register_cpsr2;
#endif

#if HAVE_dada
#include "dsp/DADABuffer.h"
static dsp::File::Register::Enter<dsp::DADABuffer> dada_buffer;
#endif

#if HAVE_dummy
#include "dsp/DummyFile.h"
static dsp::File::Register::Enter<dsp::DummyFile> register_dummy;
#endif

#if HAVE_fadc
#include "dsp/FadcFile.h"
static dsp::File::Register::Enter<dsp::FadcFile> register_fadc;
#endif

#if HAVE_fits
#include "dsp/FITSFile.h"
static dsp::File::Register::Enter<dsp::FITSFile> register_fits;
#endif

#if HAVE_gmrt
#include "dsp/GMRTFile.h"
static dsp::File::Register::Enter<dsp::GMRTFile> register_gmrt;
#include "dsp/GMRTBinaryFile.h"
static dsp::File::Register::Enter<dsp::GMRTBinaryFile> register_gmrt_b;
#include "dsp/GMRTFilterbankFile.h"
static dsp::File::Register::Enter<dsp::GMRTFilterbankFile> gmrt_fb;
#endif

#if HAVE_guppi
#include "dsp/GUPPIFile.h"
static dsp::File::Register::Enter<dsp::GUPPIFile> register_guppi;
#if HAVE_GUPPI_DAQ
#include "dsp/GUPPIBuffer.h"
static dsp::File::Register::Enter<dsp::GUPPIBuffer> guppi_buffer;
#endif
#endif

#if HAVE_lbadr
#include "dsp/SMROFile.h"
static dsp::File::Register::Enter<dsp::SMROFile> register_lbadr;
#endif

#if HAVE_lbadr64
#include "dsp/LBADR64_File.h"
static dsp::File::Register::Enter<dsp::LBADR64_File> register_lbadr64;
#endif

#if HAVE_lofar_dal
#include "dsp/LOFAR_DALFile.h"
static dsp::File::Register::Enter<dsp::LOFAR_DALFile> register_lofar_dal;
#endif

#if HAVE_lump
#include "dsp/LuMPFile.h"
static Registry::List<dsp::File>::Enter<dsp::LuMPFile> register_lump;
#endif

#if HAVE_mark4
#include "dsp/Mark4File.h"
static dsp::File::Register::Enter<dsp::Mark4File> register_mark4;
#endif

#if HAVE_mark5
#include "dsp/Mark5File.h"
static dsp::File::Register::Enter<dsp::Mark5File> register_mark5;
#endif

#if HAVE_maxim
#include "dsp/MaximFile.h"
static dsp::File::Register::Enter<dsp::MaximFile> register_maxim;
#endif

#if HAVE_spda1k
#include "dsp/spda1k_File.h"
static dsp::File::Register::Enter<dsp::SPDA1K_File> register_spda1k;
#endif

#if HAVE_mini
#include "dsp/MiniFile.h"
static dsp::File::Register::Enter<dsp::MiniFile> register_minifile;
#endif

#if HAVE_mwa
#include "dsp/MWAFile.h"
static dsp::File::Register::Enter<dsp::MWAFile> file_register_mwa;
#endif

#if HAVE_pdev
#include "dsp/PdevFile.h"
static dsp::File::Register::Enter<dsp::PdevFile> register_pdev;
#endif

#if HAVE_pmdaq
#include "dsp/PMDAQFile.h"
static dsp::File::Register::Enter<dsp::PMDAQFile> register_pmdaq;
#endif

#if HAVE_puma
#include "dsp/PuMaFile.h"
static dsp::File::Register::Enter<dsp::PuMaFile> register_puma;
#endif

#if HAVE_puma2
#include "dsp/PuMa2File.h"
static dsp::File::Register::Enter<dsp::PuMa2File> register_puma2;
#endif

#if HAVE_s2
#include "dsp/S2File.h"
static dsp::File::Register::Enter<dsp::S2File> register_s2;
#endif

#if HAVE_sigproc
#include "dsp/SigProcFile.h"
static dsp::File::Register::Enter<dsp::SigProcFile> register_sigproc;
#endif

#if HAVE_spigot
#include "dsp/SpigotFile.h"
static dsp::File::Register::Enter<dsp::SpigotFile> register_spigot;
#endif

#if HAVE_vdif
#include "dsp/VDIFFile.h"
static dsp::File::Register::Enter<dsp::VDIFFile> register_vdif;
#endif

#if HAVE_wapp
#include "dsp/WAPPFile.h"
static dsp::File::Register::Enter<dsp::WAPPFile> register_wapp;
#endif

// ///////////////////////////////////////////////////////////////////////////
//
// The rest of these need work or seem to have disappeared
//
// ///////////////////////////////////////////////////////////////////////////

//Tej's new CPSR2 8 bit File Format:
#if HAVE_CPSR2_8bit
#include "dsp/EightBitFile.h"
static dsp::File::Register::Enter<dsp::EightBitFile> register_eightbitcpsr2;
#endif

#if HAVE_vsib
#include "dsp/VSIBFile.h"
static dsp::File::Register::Enter<dsp::VSIBFile> register_vsib;
#endif

#if HAVE_DUMBLBA
#include "dsp/DumbLBAFile.h"
static dsp::File::Register::Enter<dsp::Dumb_LBAFile> file_register_dumblba;
#endif

#if HAVE_k5
#include "dsp/K5File.h"
static dsp::File::Register::Enter<dsp::K5File> register_k5;
#endif

/*! MultFile is built in and slow - it tries to parse entire binary files */
#include "dsp/MultiFile.h"
static dsp::File::Register::Enter<dsp::MultiFile> multifile;

#include "dsp/Multiplex.h"
static dsp::File::Register::Enter<dsp::Multiplex> multiplex;



/*
  get_register is defined here to ensure that this file is linked
*/
dsp::File::Register& dsp::File::get_register ()
{
  return Register::get_registry();
}



/* NOTE: DO_NOT change the order without testing all file types.
   S2 should remain last as it doesn't do proper "is_valid()" tests.
   Ensure that anything you add does do proper "is_valid()" tests
*/

// Based on Makefile.local, backends.h #defines the DSP_Backend macros
#include "backends.h"
#include "Error.h"

#if DSP_CPSR2
#include "dsp/CPSR2File.h"
static Registry::List<dsp::File>::Enter<dsp::CPSR2File> register_cpsr2;
#endif

#if DSP_CPSR
#include "dsp/CPSRFile.h"
static Registry::List<dsp::File>::Enter<dsp::CPSRFile> register_cpsr;
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


Registry::List<dsp::File> dsp::File::registry;

//! Return a pointer to a new instance of the appropriate sub-class
dsp::File* dsp::File::create (const char* filename)
{ 
  try {

    if (verbose) cerr << "File::create with " << registry.size() 
		      << " registered sub-classes" << endl;

    for (unsigned ichild=0; ichild < registry.size(); ichild++){
      if ( registry[ichild]->is_valid (filename) ) {
	
	File* child = registry.create (ichild);
	
	child-> open( filename );
	
	return child;
	
      }
    }
    
  } catch (Error& error) {
    throw error += "File::create";
  }
  
  throw Error (FileNotFound, "File::create", "%s not valid", filename);
}






/* NOTE: DO_NOT change the order without testing all file types.
   S2 should remain last as it doesn't do proper "is_valid()" tests.
   Ensure that anything you add does do proper "is_valid()" tests
*/

// Based on Makefile.local, backends.h #defines the DSP_Backend macros

#include "dsp/File.h"

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

//! Return a pointer to a new instance of the appropriate sub-class
dsp::File* dsp::File::create (const char* filename)
{ 
  // check if file can be opened for reading
  FILE* fptr = fopen (filename, "r");
  if (!fptr) throw Error (FailedSys, "dsp::File::create",
			  "cannot open '%s'", filename);
  fclose (fptr);

  try {

    if (verbose) cerr << "dsp::File::create with " << registry.size() 
		      << " registered sub-classes" << endl;

    for (unsigned ichild=0; ichild < registry.size(); ichild++) {

      if ( registry[ichild]->is_valid (filename) ) {
	
	File* child = registry.create (ichild);
	
	child-> open( filename );
	
	return child;
	
      }

    }
    
  } catch (Error& error) {
    throw error += "dsp::File::create";
  }
  
  string msg = filename;

  msg += " not a recognized file format\n\tRegistered Formats: ";

  for (unsigned ichild=0; ichild < registry.size(); ichild++)
    msg += registry[ichild]->get_name() + " ";

  throw Error (InvalidParam, "dsp::File::create", msg);

}





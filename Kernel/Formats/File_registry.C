#include "Error.h"

#include "dsp/CPSR2File.h"
#include "dsp/CPSRFile.h"
#include "dsp/S2File.h"
#include "dsp/PMDAQFile.h"
//#include "dsp/DigiFile.h"

Registry::List<dsp::File> dsp::File::registry;

//DO_NOT change the order without testing all file types
//S2 should remain last as it doesn't do proper "is_valid()" tests
//Ensure that anything you add does do proper "is_valid()" tests

static Registry::List<dsp::File>::Enter<dsp::CPSR2File> cpsr2;
static Registry::List<dsp::File>::Enter<dsp::CPSRFile>  cpsr;
static Registry::List<dsp::File>::Enter<dsp::PMDAQFile> pmdaq;
static Registry::List<dsp::File>::Enter<dsp::S2File>    s2;

//static Registry::List<dsp::File>::Enter<dsp::DigiFile> digifile;

//! Return a pointer to a new instance of the appropriate sub-class
dsp::File* dsp::File::create (const char* filename)
{ 
  try {

    if (verbose) cerr << "File::create with " << registry.size() 
		      << " registered sub-classes" << endl;

    for (unsigned ichild=0; ichild < registry.size(); ichild++)
      if ( registry[ichild]->is_valid (filename) ) {

	File* child = registry.create (ichild);
	child-> open( filename );
	return child;

      }
  } catch (Error& error) {
    throw error += "File::create";
  }

  throw Error (FileNotFound, "File::create", "%s not valid", filename);
}

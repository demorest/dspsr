#include "Error.h"

#include "dsp/CPSR2File.h"
#include "dsp/CPSRFile.h"
#include "dsp/S2File.h"
#include "dsp/PMDAQFile.h"
#include "dsp/CoherentFBFile.h"
#include "dsp/BitSeriesFile.h"

//#include "dsp/DigiFile.h"

Registry::List<dsp::File> dsp::File::registry;

//DO_NOT change the order without testing all file types
//S2 should remain last as it doesn't do proper "is_valid()" tests
//Ensure that anything you add does do proper "is_valid()" tests

static Registry::List<dsp::File>::Enter<dsp::CPSR2File>      cpsr2;
static Registry::List<dsp::File>::Enter<dsp::CPSRFile>       cpsr;
static Registry::List<dsp::File>::Enter<dsp::PMDAQFile>      pmdaq;
static Registry::List<dsp::File>::Enter<dsp::CoherentFBFile> coherentfb;
static Registry::List<dsp::File>::Enter<dsp::BitSeriesFile>  bitseries_file;
static Registry::List<dsp::File>::Enter<dsp::S2File>         s2;

//static Registry::List<dsp::File>::Enter<dsp::DigiFile> digifile;

//! Return a pointer to a new instance of the appropriate sub-class
dsp::File* dsp::File::create (const char* filename)
{ 
  try {

    if (verbose) cerr << "File::create with " << registry.size() 
		      << " registered sub-classes" << endl;

    for (unsigned ichild=0; ichild < registry.size(); ichild++){
      //fprintf(stderr,"dsp::File::create() trying ichild %d\n",ichild);
      if ( registry[ichild]->is_valid (filename) ) {
	//fprintf(stderr,"dsp::File::create() successful with ichild %d\n",ichild);

	File* child = registry.create (ichild);
	if( verbose ){
	  if( dynamic_cast<CPSR2File*>(child) )
	    fprintf(stderr,"dsp::File::create() created instance of a CPSR2File\n");
	  else if( dynamic_cast<CPSRFile*>(child) )
	    fprintf(stderr,"dsp::File::create() created instance of a CPSRFile\n");
	  else if( dynamic_cast<PMDAQFile*>(child) )
	    fprintf(stderr,"dsp::File::create() created instance of a PMDAQFile\n");
	  else if( dynamic_cast<CoherentFBFile*>(child) )
	    fprintf(stderr,"dsp::File::create() created instance of a CoherentFBFile\n");
	  else if( dynamic_cast<BitSeriesFile*>(child) )
	    fprintf(stderr,"dsp::File::create() created instance of a BitSeriesFile\n");
	  else if( dynamic_cast<S2File*>(child) )
	    fprintf(stderr,"dsp::File::create() created instance of a S2File\n");
	  else
	    fprintf(stderr,"dsp::File::create() created unknown instantiation!\n");
	}

	//fprintf(stderr,"dsp::File::create() calling child->open()\n");
	child-> open( filename );
	//fprintf(stderr,"dsp::File::create() called child->open().  Returning.\n");

	return child;

      }
    }

  } catch (Error& error) {
    throw error += "File::create";
  }

  throw Error (FileNotFound, "File::create", "%s not valid", filename);
}





/***************************************************************************
 *
 *   Copyright (C) 2003 by Haydon Knight
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include <iostream>
#include <unistd.h>

#include "dsp/IOManager.h"
#include "dsp/Input.h"
#include "dsp/Unpacker.h"
#include "dsp/WeightedTimeSeries.h"
#include "dsp/PhaseSeries.h"
#include "dsp/Detection.h"
#include "dsp/Fold.h"
#include "dsp/Archiver.h"

#include "Pulsar/BasebandArchive.h"

#include "psrephem.h"
#include "strutil.h"
#include "dirutil.h"
#include "Error.h"

using namespace std;

static char* args = "b:B:d:f:F:hm:M:n:N:o:p:P:t:v:V";

void usage ()
{
  cout << "test_Archiver - test phase coherent dedispersion kernel\n"
    "Usage: test_Archiver [" << args << "]\n"
    " -b nbin        Number of bins [1024]\n"
    " -B block_size  (in number of time samples) [512*1024]\n"
    " -d dm          DM to put in archive header [use psrinfo]\n"
    " -f filename    Filename [required]\n"
    " -m metafile    Metafile of pulsar names ['-m' or '-P']\n"
    " -n [1|2|4]     ndim kludge when forming stokes [4]\n"
    " -N nfiles      Truncate the file list to this many files [don't truncate]\n"
    " -o nfiles      Start at this file number [0]\n"
    " -p period      Fold at this period in seconds (for search data) [use ephemerides]\n"
    " -P Pulsar      Over-ride pulsar name with this name [not done]\n"
    " -t blocks      (stop before the end of the file) [goto end of file]\n"
    " -v <arg>       Verbose mode where <arg> is one of:\n"
    "                      'debug' [false]\n"
    "                      'Archive [false]\n"
    "                      'Archiver [false]\n"
    "                      'Input' [false]\n"
    "                      'Operation' [false]\n"
    "                      'Observation' [false]\n"
    "                      'main' [false]\n"
    " -V             Full verbosity [false]\n"
       << endl;
  exit(0);
}

void get_pulsarnames(vector<string>& pulsarnames, char* optarg);

int main (int argc, char** argv) 

{ try {
  Error::verbose = true;
  dsp::Operation::record_time = true;

  vector<string> pulsarnames;
  vector<string> filenames;

  string pulsar_name;

  bool verbose = false;
  bool debug = false;

  // number of time samples loaded from file at a time
  
  int block_size = 1024*1024;
  int blocks = 0;
  int ndim = 4;
  int nbin = 1024;

  bool dm_supplied = false;
  float header_dm = 0.0;
  
  bool use_fold_period = false;
  double fold_period = 0.0;


  int c;
  while ((c = getopt(argc, argv, args)) != -1){
    switch (c) {

    case 'b':
      nbin = atoi (optarg);
      break;
    case 'B':
      block_size = atoi (optarg);
      break;
    case 'd':
      dm_supplied = true;
      header_dm = atof(optarg);
      break;
    case 'f':
      filenames.push_back( optarg );
      break;
    case 'h':
      usage();
    case 'm':
      get_pulsarnames( pulsarnames, optarg);
      break;
    case 'n':
      ndim = atoi (optarg);
      break;
    case 'p':
      use_fold_period = true;
      fold_period = atof(optarg);
      break;
    case 'P':
      pulsarnames.push_back(optarg);
      break;
   case 't':
      blocks = atoi (optarg);
      break;
    case 'v':
      {
	string arg = optarg;
	if( arg=="Input" )
	  dsp::Input::verbose = true;
	else if( arg=="debug" )
	  debug = true;
	else if( arg=="Operation" )
	  dsp::Operation::verbose = true;
	else if( arg=="Observation" )
	  dsp::Observation::verbose = true;
	else if( arg=="main" )
	  verbose = true;
	else if( arg=="Archive" )
	  Pulsar::Archive::set_verbosity(3);
	else if( arg=="Archiver" )
	  dsp::Archiver::verbose = true;
	else{
	  fprintf(stderr,"Please follow your '-v' option with one of 'Ready', 'Input', 'Operation', 'Observation'\n");
	  exit(-1);
	}
      }
      break;
    case 'V':
      Pulsar::Archive::set_verbosity (3);
      dsp::Observation::verbose = true;
      dsp::Operation::verbose = true;
      dsp::Input::verbose = true;
      dsp::Archiver::verbose = true;
      debug = true;
      verbose = true;
      break;

    default:
      cerr << "invalid param '" << c << "'" << endl;
    }
  }
 
  if( filenames.empty() )
    usage ();
  if (pulsarnames.empty())
    usage ();



  if( verbose )
    fprintf(stderr,"Will work with %d files starting with '%s'\n",
	    filenames.size(), filenames.front().c_str() );

  //
  // Finished parsing- start setting up
  //
  for (unsigned int ifile = 0; ifile < filenames.size(); ifile++) {
    
    dsp::WeightedTimeSeries* voltages = new dsp::WeightedTimeSeries;
    dsp::WeightedTimeSeries* combined_voltages = voltages;
    vector<dsp::PhaseSeries*> psrprofiles;
    vector<dsp::Fold*> folders;
    vector<Pulsar::BasebandArchive*> archives;
     

    if (pulsarnames.size()>0) {
       for(unsigned int i=0;i<pulsarnames.size();i++) {
         psrprofiles.push_back(new dsp::PhaseSeries);
         folders.push_back(new dsp::Fold);
         folders.back()->set_input(combined_voltages);
         folders.back()->set_output(psrprofiles.back());
         folders.back()->set_nbin (nbin);
         archives.push_back(new Pulsar::BasebandArchive);
       }
    }
    
    // Loader
    
    dsp::IOManager manager;
    manager.open( filenames.front() );
    manager.get_input()->set_block_size (block_size);


    if (verbose)
      cerr << "Creating Archiver instance" << endl;
    dsp::Archiver archiver;


    // Detector
    dsp::Detection* detect = new dsp::Detection;
  
    detect->set_output_state (Signal::Coherence);
    detect->set_output_ndim (ndim);
  
    detect->set_input (combined_voltages);
    detect->set_output (combined_voltages);

    int block=0;

    while (!manager.get_input()->eod()) {

      manager.load (voltages);
      
      if( !combined_voltages->get_detected() )
	detect->operate ();

      if (pulsarnames.size()>0) {
        for (unsigned int i=0;i<pulsarnames.size();i++){	
          combined_voltages->set_source(pulsarnames[i].c_str());
          if (verbose)
            cerr << "have set source to '" << pulsarnames[i].c_str() <<"'" << endl;

          folders[i]->operate();
        }
      }
      block++;
      if (verbose)
        cerr << "finished " << block << " blocks\r";
      if (block == blocks) break;
    }
    
    if (verbose)
      cerr << "end of data file " << filenames[ifile] << endl;
    
    if (pulsarnames.size() > 0) {
      for (unsigned int i=0;i<pulsarnames.size();i++) { 
        archiver.set (&*archives[i], &*psrprofiles[i]);
    
        if( dm_supplied ){
          if (verbose)
            fprintf(stderr,"A dm was specified on command line\n");
          archives[i]->set_dispersion_measure( header_dm );
        }
    
        else if( folders[i]->get_pulsar_ephemeris() ){
          if (verbose)
            fprintf(stderr,"No dm was specified on command line\n");

	  const psrephem* eph = dynamic_cast<const psrephem*>
	    (folders[i]->get_pulsar_ephemeris());

	  if (eph)
	    archives[i]->set_dispersion_measure(eph->get_dm());
        }
        else{
          if (verbose)
            fprintf(stderr,"No DM or ephemeris was specified.  Header DM will be zero\n");
          archives[i]->set_dispersion_measure( header_dm );
        }
        string extn  ;
        extn = "_" + pulsarnames[i]; 
        string band;

        if (archives[i]->get_centre_frequency() >= 1390) {
          band = "m";
        }
        else if (archives[i]->get_centre_frequency()< 1380) {
          band = "n";
        }
        
        string filename = band + psrprofiles[i]->get_default_id () + extn + ".ar";
 
        if (verbose) cerr << "Unloading archive: " << filename<< endl;
        
        archives[i]->Archive::unload (filename);
          
      } 
    }
  }

  } catch (Error& error) {
    cerr << "Error thrown: " << error << endl;
    return -1;
  } catch (string& error) {
    cerr << "exception thrown: " << error << endl;
    return -1;
  } catch (...) {
    cerr << "unknown exception thrown." << endl;
    return -1;
  } 

  return 0;
}

void get_pulsarnames(vector<string>& pulsarnames, char* optarg){
  FILE* fptr = fopen(optarg,"r");
  char cfilename[1024];
  while( fscanf(fptr,"%s\n",cfilename) != EOF){
    pulsarnames.push_back( cfilename );
  }
  fclose(fptr);
}

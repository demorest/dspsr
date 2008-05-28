/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
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
    " -f filename    Filename ['-m' or '-f' req]\n"
    " -F buddy_file  Filename of buddy channel [1 band- no buddy]\n"
    " -m metafile    Metafile of filenames ['-f' or '-m' req]\n"
    " -M buddy_meta  Buddy metafile of other band's filenames [1 band- no buddy]\n"
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
}

void get_filenames(vector<string>& filenames, char* optarg);

int main (int argc, char** argv) 

{ try {
  Error::verbose = true;
  dsp::Operation::record_time = true;

  vector<string> filenames;
  vector<string> buddy_filenames;
  bool want_buddies = false;

  bool verbose = false;
  bool debug = false;

  // number of time samples loaded from file at a time
  int block_size = 512*1024;
  int blocks = 0;
  int ndim = 4;
  int nbin = 1024;

  bool dm_supplied = false;
  float header_dm = 0.0;
  
  bool use_fold_period = false;
  double fold_period = 0.0;

  string pulsar_name;

  unsigned file_offset = 0;
  int truncate_files = -1;

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
    case 'F':
      want_buddies = true;
      buddy_filenames.push_back( optarg );
      break;
    case 'h':
      usage();
      return 0;
    case 'm':
      get_filenames( filenames, optarg);
      break;
    case 'M':
      want_buddies = true;
      get_filenames( buddy_filenames, optarg);
      break;
    case 'n':
      ndim = atoi (optarg);
      break;
    case 'N':
      truncate_files = atoi(optarg);
      break;
    case 'o':
      file_offset = atoi(optarg);
      break;
    case 'p':
      use_fold_period = true;
      fold_period = atof(optarg);
      break;
    case 'P':
      pulsar_name = optarg;
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
  
  if( want_buddies && buddy_filenames.size()!=filenames.size() ){
    unsigned min_sz = min(filenames.size(),buddy_filenames.size());
    
    fprintf(stderr,"You supplied %d filenames and %d buddy filenames.  Will use just %d of %s.\n",
	    filenames.size(),buddy_filenames.size(),min_sz,
	    filenames.size()<buddy_filenames.size()?"buddy_filenames":"filenames");

    if( buddy_filenames.size()>filenames.size() )
      buddy_filenames.resize( filenames.size() );
    else
      filenames.resize( buddy_filenames.size() );

  }

  if( filenames.empty() ) {
    usage ();
    return -1;
  }

  if( file_offset>=filenames.size() )
    throw Error(InvalidParam,"main()",
		"Your file offset (%d) is >= than the number of files (%d)\n",
		file_offset,filenames.size());

  filenames.erase(filenames.begin(),filenames.begin()+file_offset);
  if( want_buddies )
    buddy_filenames.erase(buddy_filenames.begin(),buddy_filenames.begin()+file_offset);

  if( truncate_files>0 && unsigned(truncate_files)<filenames.size() ){
    filenames.resize( truncate_files );
    if( want_buddies )
      buddy_filenames.resize( truncate_files );
  }

  if( verbose )
    fprintf(stderr,"Will work with %d files starting with '%s'\n",
	    filenames.size(), filenames.front().c_str() );

  //
  // Finished parsing- start setting up
  //

  dsp::WeightedTimeSeries* voltages = new dsp::WeightedTimeSeries;
  dsp::WeightedTimeSeries* buddy_voltages = new dsp::WeightedTimeSeries;
  dsp::WeightedTimeSeries* combined_voltages = voltages;
  if( want_buddies )
    combined_voltages = new dsp::WeightedTimeSeries;

  dsp::PhaseSeries profiles;

  // Loader
  fprintf(stderr,"Creating instance of IOManager\n");
  dsp::IOManager manager;
  fprintf(stderr,"block size done.  calling manager.open()\n");
  manager.open( filenames.front() );

  dsp::IOManager buddy_manager;

  // Detector
  dsp::Detection* detect = new dsp::Detection;
  
  detect->set_output_state (Signal::Coherence);
  detect->set_output_ndim (ndim);
  
  detect->set_input (combined_voltages);
  detect->set_output (combined_voltages);
 
  if (verbose)
    cerr << "Creating Fold instance" << endl;
  dsp::Fold fold;

  fold.set_nbin (nbin);
  if( use_fold_period )
    fold.set_folding_period( fold_period );
  fold.set_input( combined_voltages );
  fold.set_output (&profiles);

  dsp::Archiver archiver;
  Pulsar::BasebandArchive archive;

  for (unsigned ifile=0; ifile < filenames.size(); ifile++) {
    
    if (verbose)
      cerr << "opening data file " << filenames[ifile] << endl;

    manager.open (filenames[ifile]);
    manager.get_input()->set_block_size (block_size);

    if( want_buddies ) {
      buddy_manager.open (buddy_filenames[ifile]);
      buddy_manager.get_input()->set_block_size (block_size);
    }

    if (verbose)
      cerr << "data file " << filenames[ifile] << " opened" << endl;

    if (manager.get_info()->get_state() == Signal::Nyquist)
      detect->set_output_state (Signal::Intensity);

    if (manager.get_info()->get_state() == Signal::Analytic)
      detect->set_output_state (Signal::Coherence);

    profiles.zero ();

    fold.prepare ( manager.get_info() );

    int block=0;

    while (!manager.get_input()->eod()) {

      manager.load (voltages);
      fprintf(stderr,"After load voltages->ndat="UI64"\n",
		voltages->get_ndat());

      if( want_buddies )
	buddy_manager.load( buddy_voltages );

      if( want_buddies ){
	fprintf(stderr,"before combining bands voltages->ndat="UI64"\n",
		voltages->get_ndat());
	
	// combined_voltages just has the null weightings so far, so using this TimeSeries method is ok.
	throw Error(InvalidState,"main()",
		    "hack_together now deprecated.  If anyone still uses this code please yell out.");

	//combined_voltages->hack_together(voltages,buddy_voltages);
      }

      if( pulsar_name!=string("") )
	combined_voltages->set_source( pulsar_name );
      
      if (debug) {
	cerr << "check " << combined_voltages->get_state()
	     << " voltages block " << block << endl;
	combined_voltages->check();
      }
      
      if( !combined_voltages->get_detected() )
	detect->operate ();
	
      fold.operate ();
      
      block++;
      
      if (debug) {
	float range = block*block_size / nbin;
	
	cerr << "check " << profiles.get_state()
	     << " profiles block " << block << " +/-" << range << endl;
	profiles.check (-range, range);
      }
      
      cerr << "finished " << block << " blocks\r";
      if (block == blocks) break;
    }
    
    if (verbose)
      cerr << "end of data file " << filenames[ifile] << endl;
    
    cerr << "Time spent converting data: " 
	 << manager.get_unpacker()->get_total_time() << " seconds" << endl;
    
    cerr << "Time spent detecting" << ndim << " data: " 
	 << detect->get_total_time() << " seconds" << endl;
    
    cerr << "Time spent folding" << ndim << " data: " 
	 << fold.get_total_time() << " seconds" << endl;
    
    archiver.set (&archive, &profiles);
    
    fprintf(stderr,"In main(), but out of archiver.set()\n");
    
    if( dm_supplied ){
      fprintf(stderr,"A dm was specified on command line\n");
      archive.set_dispersion_measure( header_dm );
    }
    else if( fold.get_pulsar_ephemeris() ){
      fprintf(stderr,"No dm was specified on command line\n");
      archive.set_dispersion_measure(fold.get_pulsar_ephemeris()->get_dm());
    }
    else{
      fprintf(stderr,"No DM or ephemeris was specified.  Header DM will be zero\n");
      archive.set_dispersion_measure( header_dm );
    }
    string extn  ;
    if (pulsar_name != "") {
      extn = "_" + pulsar_name; 
    }
    

    string filename = profiles.get_default_id () + extn + ".ar";
 
    if (verbose) cerr << "Unloading archive: " << filename<< endl;
      fprintf(stderr,"Directly before archive.unload()\n");
    archive.Archive::unload (filename);
    
    
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
  
 fprintf(stderr,"biyee!\n"); 
 exit(0);

}

void get_filenames(vector<string>& filenames, char* optarg){
  FILE* fptr = fopen(optarg,"r");
  char cfilename[1024];
  while( fgets(cfilename,1023,fptr) ){
    filenames.push_back( cfilename );
    h_backchomp( filenames.back() );
    if( filenames.back()==string("") ){
      filenames.pop_back();
      continue;
    }
  }
  fclose(fptr);
}













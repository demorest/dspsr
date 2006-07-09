/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include <iostream>
#include <unistd.h>

#include "tempo++.h"
#include "string_utils.h"
#include "dirutil.h"
#include "Error.h"

#include "dsp/IOManager.h"
#include "dsp/Input.h"
#include "dsp/Unpacker.h"
#include "dsp/TimeSeries.h"
#include "dsp/BitSeries.h"
#include "dsp/PhaseSeries.h"
#include "dsp/Detection.h"
#include "dsp/Fold.h"

static char* args = "b:n:op:t:vV";

void usage ()
{
  cout << "test_Fold - test phase coherent dedispersion kernel\n"
    "Usage: test_Fold [" << args << "] file1 [file2 ...] \n"
       << endl;
}

int main (int argc, char** argv) 

{ try {
  Error::verbose = true;
  Error::complete_abort = true;

  char* metafile = 0;
  bool verbose = false;

  // number of time samples loaded from file at a time
  int block_size = 512*1024;
  int blocks = 0;
  int ndim = 1;
  int nbin = 1024;
  Signal::State output_state = Signal::Coherence;
  bool inplace_detection = true;

  int c;
  while ((c = getopt(argc, argv, args)) != -1)
    switch (c) {

    case 'V':
      Tempo::verbose = true;
      dsp::Operation::verbose = true;
      dsp::Input::verbose = true;
      dsp::Observation::verbose = true;
    case 'v':
      verbose = true;
      break;

    case 'b':
      nbin = atoi (optarg);
      break;

    case 't':
      blocks = atoi (optarg);
      break;

    case 'n':
      ndim = atoi (optarg);
      break;

    case 'o': inplace_detection = false; break;

    case 'p':
      {
	unsigned npol = atoi(optarg);
	if( npol==1 ) output_state = Signal::Intensity;
	if( npol==4 ) output_state = Signal::Coherence;
	break;
      }

    default:
      cerr << "invalid param '" << c << "'" << endl;
    }

  vector <string> filenames;

  if (metafile)
    stringfload (&filenames, metafile);
  else 
    for (int ai=optind; ai<argc; ai++)
      dirglob (&filenames, argv[ai]);

  if (filenames.size() == 0) {
    usage ();
    return -1;
  }

  if (verbose)
    cerr << "Creating TimeSeries instance" << endl;
  dsp::TimeSeries voltages;

  if (verbose)
    cerr << "Creating PhaseSeries instance" << endl;
  dsp::PhaseSeries profiles;

  
  if (verbose)
    cerr << "Creating IOManager instance" << endl;
  dsp::IOManager manager;
  
  if (verbose)
    cerr << "Creating Detection instance" << endl;
  dsp::Detection detect;

  detect.set_output_state( output_state );
  detect.set_output_ndim (ndim);
  detect.set_input (&voltages);
  if( inplace_detection )
    detect.set_output (&voltages);
  else
    detect.set_output (new dsp::TimeSeries);

  if (verbose)
    cerr << "Creating Fold instance" << endl;
  dsp::Fold fold;

  fold.set_nbin (nbin);

  fold.set_input (detect.get_output());
  fold.set_output (&profiles);

  dsp::Operation::record_time = true;

  for (unsigned ifile=0; ifile < filenames.size(); ifile++) try {

    if (verbose)
      cerr << "opening data file " << filenames[ifile] << endl;

    manager.open (filenames[ifile]);
    manager.get_input()->set_block_size (block_size);

    if (verbose)
      cerr << "data file " << filenames[ifile] << " opened" << endl;

    profiles.zero ();

    fold.prepare ( manager.get_info() );

    int block=0;

    while (!manager.get_input()->eod()) {

      manager.load (&voltages);

      detect.operate ();

      fold.operate ();

      block++;
      if (blocks && block==blocks) {
	cerr << "finished " << blocks << " blocks" << endl;
	break;
      }
    }

    if (verbose)
      cerr << "end of data file " << filenames[ifile] << endl;

    cerr << "Time spent converting data: " 
	 << manager.get_unpacker()->get_total_time() << " seconds" << endl;

    cerr << "Time spent detecting" << ndim << " data: " 
	 << detect.get_total_time() << " seconds" << endl;

    cerr << "Time spent folding" << ndim << " data: " 
	 << fold.get_total_time() << " seconds" << endl;
  }
  catch (string& error) {
    cerr << error << endl;
  }

  fprintf(stderr,"biyee!\n");
  return 0;
}

catch (Error& error) {
  cerr << "Error thrown: " << error << endl;
  return -1;
}

catch (string& error) {
  cerr << "exception thrown: " << error << endl;
  return -1;
}

catch (...) {
  cerr << "unknown exception thrown." << endl;
  return -1;
}
 fprintf(stderr,"At end of main()\n"); 
}

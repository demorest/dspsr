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
#include "dsp/TimeSeries.h"
#include "dsp/Detection.h"
#include "dsp/Unpacker.h"

#include "strutil.h"
#include "dirutil.h"

using namespace std;

static char* args = "b:n:vV";

void usage ()
{
  cout << "test_Detection - test phase coherent dedispersion kernel\n"
    "Usage: test_Detection [" << args << "] file1 [file2 ...] \n"
       << endl;
}

int main (int argc, char** argv) 

{ try {

  char* metafile = 0;
  bool verbose = false;

  // number of time samples loaded from file at a time
  int block_size = 512*1024;
  int blocks = 0;
  int ndim = 1;

  int c;
  while ((c = getopt(argc, argv, args)) != -1)
    switch (c) {

    case 'V':
      dsp::Operation::verbose = true;
      dsp::Input::verbose = true;
    case 'v':
      verbose = true;
      break;

    case 'b':
      blocks = atoi (optarg);
      break;

    case 'n':
      ndim = atoi (optarg);
      break;

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
    return 0;
  }

  if (verbose)
    cerr << "Creating TimeSeries instance" << endl;

  // input voltages (float_Stream)
  dsp::TimeSeries voltages;

  if (verbose)
    cerr << "Creating IOManager instance" << endl;

  // interface manages the creation of data loading and converting classes
  dsp::IOManager manager;

  if (verbose)
    cerr << "Creating Detection instance" << endl;

  dsp::Detection detect;
  detect.set_output_state (Signal::Coherence);
  detect.set_output_ndim (ndim);
  detect.set_input (&voltages);
  detect.set_output (&voltages);
  
  dsp::Operation::record_time = true;

  for (unsigned ifile=0; ifile < filenames.size(); ifile++) try {

    if (verbose)
      cerr << "opening data file " << filenames[ifile] << endl;

    manager.open (filenames[ifile]);
    manager.get_input()->set_block_size (block_size);

    if (verbose)
      cerr << "data file " << filenames[ifile] << " opened" << endl;

    int block=0;

    while (!manager.get_input()->eod()) {

      manager.load (&voltages);

      detect.operate ();

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
  }
  catch (string& error) {
    cerr << error << endl;
  }

  
  return 0;
}


catch (string& error) {
  cerr << "exception thrown: " << error << endl;
  return -1;
}

catch (...) {
  cerr << "exception thrown: " << endl;
  return -1;
}

}

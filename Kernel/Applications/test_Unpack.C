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

#include "strutil.h"
#include "dirutil.h"
#include "Error.h"

using namespace std;

static char* args = "B:t:vV";

void usage ()
{
  cout << "test_Unpack - test phase coherent dedispersion kernel\n"
    "Usage: test_Unpack [" << args << "] file1 [file2 ...] \n"
    " -B block_size  (in number of time samples)\n"
    " -t blocks      (stop before the end of the file)\n"
       << endl;
}

int main (int argc, char** argv) 

{ try {

  char* metafile = 0;
  bool verbose = false;

  // number of time samples loaded from file at a time
  int block_size = 512*1024;
  int blocks = 0;

  int c;
  while ((c = getopt(argc, argv, args)) != -1)
    switch (c) {

    case 'V':
      dsp::Observation::verbose = true;
      dsp::Operation::verbose = true;
    case 'v':
      verbose = true;
      break;

    case 'B':
      block_size = atoi (optarg);
      break;

    case 't':
      blocks = atoi (optarg);
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
  dsp::WeightedTimeSeries voltages;

  if (verbose)
    cerr << "Creating IOManager instance" << endl;

  // Loader
  dsp::IOManager manager;

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
      
      cerr << "check voltages " << voltages.get_state() 
           << " block " << block << endl;
      voltages.check();

      block++;
      if (block == blocks)
	break;
    }

    if (verbose)
      cerr << "end of data file " << filenames[ifile] << endl;

    cerr << "Time spent converting data: " 
	 << manager.get_unpacker()->get_total_time() << " seconds" << endl;

  }
  catch (string& error) {
    cerr << error << endl;

  }

  
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

}

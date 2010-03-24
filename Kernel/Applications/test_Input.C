/***************************************************************************
 *
 *   Copyright (C) 2003 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include <iostream>
#include <unistd.h>

#include "dsp/File.h"
#include "dsp/TestInput.h"

#include "strutil.h"
#include "dirutil.h"
#include "Error.h"

using namespace std;

static char* args = "b:t:vV";

void usage ()
{
  cout << "test_Input - test time sample resolution features of Input class\n"
    "Usage: test_Input [" << args << "] file1 [file2 ...] \n"
    " -b block size  the base block size used in the test\n"
    " -t blocks      (stop before the end of the file)\n"
       << endl;
}

int main (int argc, char** argv) 

{ try {

  char* metafile = 0;
  bool verbose = false;

  int blocks = 0;
  unsigned block_size = 4096;

  int c;
  while ((c = getopt(argc, argv, args)) != -1)
    switch (c) {

    case 'V':
      dsp::Observation::verbose = true;
      dsp::Operation::verbose = true;
      MJD::verbose = true;

    case 'v':
      verbose = true;
      break;

    case 'b':
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

  dsp::TestInput test;

  Reference::To<dsp::Input> input_small;
  Reference::To<dsp::Input> input_large;

  for (unsigned ifile=0; ifile < filenames.size(); ifile++) try {

    unsigned errors = 0;

    if (verbose)
      cerr << "opening data file " << filenames[ifile] << endl;

    input_small = dsp::File::create (filenames[ifile]);
    input_large = dsp::File::create (filenames[ifile]);

    cerr << "data file " << filenames[ifile] << " opened" << endl;

    dsp::Observation* obs = input_small->get_info();
    cerr << "soure name = " << obs->get_source() << endl;

    test.runtest (input_small, input_large);

    cerr << "end of data file " << filenames[ifile] << endl << endl;

    if (!errors)
      cerr << "success: dsp::Input operates as expected with " 
	   << input_large->get_name() << " sub-class" << endl;
    else {
      cerr << "failure: dsp::Input does not operate as expected with " 
	   << input_large->get_name() << " sub-class" << endl;
      return -1;
    }

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

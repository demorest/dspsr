/***************************************************************************
 *
 *   Copyright (C) 2015 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <iostream>
#include <unistd.h>

#include "dsp/File.h"
#include "TextInterface.h"

#include "strutil.h"
#include "dirutil.h"
#include "Error.h"

using namespace std;

static char* args = "hvV";

void usage ()
{
  cout << "digihdr - prints what is known about a raw data file" << endl;
  exit (0);
}

int main (int argc, char** argv) try
{
  bool verbose = false;

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

    case 'h':
      usage ();
      break;

    default:
      cerr << "invalid param '" << c << "'" << endl;
    }

  vector <string> filenames;

  for (int ai=optind; ai<argc; ai++)
    dirglob (&filenames, argv[ai]);

  if (filenames.size() == 0)
  {
    usage ();
    return 0;
  }

  Reference::To<dsp::Input> input;

  for (unsigned ifile=0; ifile < filenames.size(); ifile++) try
  {
    if (verbose)
      cerr << "opening data file " << filenames[ifile] << endl;

    input = dsp::File::create (filenames[ifile]);

    cerr << "data file " << filenames[ifile] << " opened" << endl;

    dsp::Observation* obs = input->get_info();
    
    cout << obs->get_interface()->help(true);
  }
  catch (string& error)
  {
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


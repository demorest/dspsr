/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SigProcObservation.h"
#include "dsp/File.h"

#include "dirutil.h"
#include "Error.h"

#include <iostream>
#include <unistd.h>

using namespace std;

static char* args = "hvV";

void usage ()
{
  cout << "sigproc_header - convert dspsr input to sigproc header \n"
    "Usage: sigproc_header file1 [file2 ...] \n" << endl;
}

int main (int argc, char** argv) try 
{
  bool verbose = false;

  int c;
  while ((c = getopt(argc, argv, args)) != -1)
    switch (c) {

    case 'h':
      usage ();
      return 0;

    case 'V':
      dsp::Operation::verbose = true;
      dsp::Observation::verbose = true;
    case 'v':
      verbose = true;
      break;

    default:
      cerr << "invalid param '" << c << "'" << endl;
    }

  vector <string> filenames;
  
  for (int ai=optind; ai<argc; ai++)
    dirglob (&filenames, argv[ai]);

  if (filenames.size() == 0)
  {
    cerr << "sigproc_header: please specify a filename (-h for help)" << endl;
    return 0;
  }

  // generalized interface to input data
  Reference::To<dsp::Input> input;

  for (unsigned ifile=0; ifile < filenames.size(); ifile++) try
  {
    if (verbose)
      cerr << "sigproc_header: opening file " << filenames[ifile] << endl;

    input = dsp::File::create( filenames[ifile] );

    if (verbose)
    {
      dsp::Observation* obs = input->get_info();

      cerr << "sigproc_header: file " << filenames[ifile] << " opened" << endl;
      cerr << "Source = " << obs->get_source() << endl;
      cerr << "Frequency = " << obs->get_centre_frequency() << endl;
      cerr << "Bandwidth = " << obs->get_bandwidth() << endl;
      cerr << "Sampling rate = " << obs->get_rate() << endl;
    }

    dsp::SigProcObservation sigproc;

    sigproc.copy( input->get_info() );

    sigproc.unload( stdout );

    if (verbose)
      cerr << "end of data file " << filenames[ifile] << endl;
  }
  catch (Error& error)
  {
    cerr << error << endl;
  }
  
  return 0;
}

catch (Error& error)
{
  cerr << error << endl;
  return -1;
}

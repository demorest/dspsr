#include <iostream>
#include <unistd.h>

#include "IOManager.h"
#include "Timeseries.h"
#include "PhaseSeries.h"
#include "Detection.h"
#include "Fold.h"
#include "tempo++.h"

#include "string_utils.h"
#include "dirutil.h"
#include "Error.h"

static char* args = "b:n:t:vV";

void usage ()
{
  cout << "test_Fold - test phase coherent dedispersion kernel\n"
    "Usage: test_Fold [" << args << "] file1 [file2 ...] \n"
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
  int nbin = 1024;

  int c;
  while ((c = getopt(argc, argv, args)) != -1)
    switch (c) {

    case 'V':
      Tempo::verbose = true;
      dsp::Operation::verbose = true;
      dsp::Input::verbose = true;
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
    cerr << "Creating Timeseries instance" << endl;
  dsp::Timeseries voltages;

  if (verbose)
    cerr << "Creating PhaseSeries instance" << endl;
  dsp::PhaseSeries profiles;

  if (verbose)
    cerr << "Creating IOManager instance" << endl;
  dsp::IOManager manager;

  manager.set_block_size (block_size);
  manager.set_nsample (1024);  // ppweight

  if (verbose)
    cerr << "Creating Detection instance" << endl;
  dsp::Detection detect;

  detect.set_output_state (Signal::Coherence);
  detect.set_output_ndim (ndim);
  detect.set_input (&voltages);
  detect.set_output (&voltages);

  if (verbose)
    cerr << "Creating Fold instance" << endl;
  dsp::Fold fold;

  fold.set_nbin (nbin);
  fold.set_input (&voltages);
  fold.set_output (&profiles);

  dsp::Operation::record_time = true;

  for (unsigned ifile=0; ifile < filenames.size(); ifile++) try {

    if (verbose)
      cerr << "opening data file " << filenames[ifile] << endl;

    manager.open (filenames[ifile]);

    if (verbose)
      cerr << "data file " << filenames[ifile] << " opened" << endl;

    int block=0;

    while (!manager.eod()) {

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
	 << manager.get_converter()->get_total_time() << " seconds" << endl;

    cerr << "Time spent detecting" << ndim << " data: " 
	 << detect.get_total_time() << " seconds" << endl;

    cerr << "Time spent folding" << ndim << " data: " 
	 << fold.get_total_time() << " seconds" << endl;
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

catch (Reference::invalid& error) {
  cerr << "Reference invalid exception thrown" << endl;
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

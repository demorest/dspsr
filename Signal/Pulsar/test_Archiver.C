#include <iostream>
#include <unistd.h>

#include "dsp/IOManager.h"
#include "dsp/Unpacker.h"
#include "dsp/TimeSeries.h"
#include "dsp/PhaseSeries.h"
#include "dsp/Detection.h"
#include "dsp/Fold.h"
#include "dsp/Archiver.h"

#include "Pulsar/TimerArchive.h"

#include "string_utils.h"
#include "dirutil.h"
#include "Error.h"

static char* args = "b:B:n:t:vV";

void usage ()
{
  cout << "test_Fold - test phase coherent dedispersion kernel\n"
    "Usage: test_Fold [" << args << "] file1 [file2 ...] \n"
    " -b nbin\n"
    " -B block_size  (in number of time samples)\n"
    " -t blocks      (stop before the end of the file)\n"
    " -n [1|2|4]     ndim kludge when forming stokes\n"
       << endl;
}

int main (int argc, char** argv) 

{ try {

  char* metafile = 0;
  bool verbose = false;

  // number of time samples loaded from file at a time
  int block_size = 512*1024;
  int blocks = 0;
  int ndim = 4;
  int nbin = 1024;

  int c;
  while ((c = getopt(argc, argv, args)) != -1)
    switch (c) {

    case 'V':
      Pulsar::Archive::set_verbosity (3);
      dsp::Observation::verbose = true;
      dsp::Operation::verbose = true;
      dsp::Input::verbose = true;
      dsp::Archiver::verbose = true;
    case 'v':
      verbose = true;
      break;

    case 'b':
      nbin = atoi (optarg);
      break;

    case 'B':
      block_size = atoi (optarg);
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
    cerr << "Creating TimeSeries instance" << endl;
  dsp::TimeSeries voltages;

  if (verbose)
    cerr << "Creating PhaseSeries instance" << endl;
  dsp::PhaseSeries profiles;

  if (verbose)
    cerr << "Creating IOManager instance" << endl;

  // Loader
  dsp::IOManager manager;
  manager.set_block_size (block_size);  

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

  if (verbose)
    cerr << "Creating Archiver instance" << endl;
  dsp::Archiver archiver;

  if (verbose)
    cerr << "Creating TimerArchive instance" << endl;
  Pulsar::TimerArchive archive;

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

      cerr << "check voltages " << voltages.get_state_as_string() 
           << " block " << block << endl;
      voltages.check();
      
      if (!voltages.get_detected())  {
        detect.operate ();
        cerr << "check voltages " << voltages.get_state_as_string()
             << " block " << block << endl;
        voltages.check();
      }

      fold.operate ();

      cerr << "check profiles " << profiles.get_state_as_string()
           << " block " << block << endl;

      block++;

      cerr << "+/-" << block*block_size << endl;
      profiles.check(-block*block_size, block*block_size);

      cerr << "finished " << block << " blocks\r";
      if (block == blocks) break;
    }

    if (verbose)
      cerr << "end of data file " << filenames[ifile] << endl;

    cerr << "Time spent converting data: " 
	 << manager.get_unpacker()->get_total_time() << " seconds" << endl;

    cerr << "Time spent detecting" << ndim << " data: " 
	 << detect.get_total_time() << " seconds" << endl;

    cerr << "Time spent folding" << ndim << " data: " 
	 << fold.get_total_time() << " seconds" << endl;

    archiver.set (&archive, &profiles);
    archive.set_dispersion_measure(fold.get_pulsar_ephemeris()->get_dm());

    string filename = profiles.get_default_id () + ".ar";

    if (verbose) cerr << "Unloading archive: " << filename<< endl;
    archive.unload (filename.c_str());

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

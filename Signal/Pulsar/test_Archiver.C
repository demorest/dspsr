#include <iostream>
#include <unistd.h>

#include "IOManager.h"
#include "Timeseries.h"
#include "PhaseSeries.h"
#include "Detection.h"
#include "Fold.h"
#include "Archiver.h"
#include "Reference.h"
#include "File.h"
#include "Unpacker.h"
#include "OneBitCorrection.h"

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
  int overlap = 0;
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
    cerr << "Creating Timeseries instance" << endl;
  dsp::Timeseries raw;
  dsp::Timeseries voltages;

  if (verbose)
    cerr << "Creating PhaseSeries instance" << endl;
  dsp::PhaseSeries profiles;

  if (verbose)
    cerr << "Creating Loader and OneBitCorrection instances" << endl;

  // Loader
  Reference::To<dsp::File> loader(dsp::File::create(filenames.front().c_str()));
  loader->open( filenames[0].c_str() );
  loader->set_block_size (block_size);  

  // OneBitCorrection
  Reference::To<dsp::Unpacker> converter(dsp::Unpacker::create(loader->get_info()));
  fprintf(stderr,"\nOut of dsp::Unpacker::create()\n");
  converter->set_input( &raw );
  converter->set_output( &voltages );
 
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

    loader->open (filenames[ifile]);

    if (verbose)
      cerr << "data file " << filenames[ifile] << " opened" << endl;

    int block=0;

    while (!loader->eod()) {

      loader->load (&raw);
      
      if(raw.get_ndat()==0){
	cerr << "Breaking\n";
	break;
      }
      converter->operate();

      if (!voltages.get_detected())
        detect.operate ();

      if (voltages.get_ndat()>0) {
	fold.operate ();
      }
      block++;
      cerr << "finished " << block << " blocks\r";
      if (block == blocks) break;
    }

    if (verbose)
      cerr << "end of data file " << filenames[ifile] << endl;

    cerr << "Time spent converting data: " 
	 << converter->get_total_time() << " seconds" << endl;

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

#include <iostream>
#include <unistd.h>

#include <cpgplot.h>

#include "dsp/TwoBitStatsPlotter.h"
#include "dsp/TwoBitCorrection.h"
#include "dsp/BitSeries.h"
#include "dsp/TimeSeries.h"
#include "dsp/IOManager.h"
#include "dsp/Input.h"
#include "Error.h"

#include "string_utils.h"
#include "dirutil.h"

static char* args = "c:n:t:vV";

void usage ()
{
  cout << "digistat - plots digitizer statistics\n"
    "Usage: digistat [" << args << "] file1 [file2 ...] \n"
    " -n <nsample>   number of samples used in estimating undigitized power\n"
    " -c <cutoff>    cutoff threshold for impulsive interference excision\n"
    " -t <threshold> sampling threshold at record time\n"
       << endl;
}

int main (int argc, char** argv) 

{ try {

  char* metafile = 0;
  bool display = true;
  bool verbose = false;

  unsigned tbc_nsample = 512;
  float tbc_cutoff = 0.0;
  float tbc_threshold = 0.0;

  int c;
  int scanned;
  while ((c = getopt(argc, argv, args)) != -1)
    switch (c) {

    case 'n':
      scanned = sscanf (optarg, "%u", &tbc_nsample);
      if (scanned != 1) {
	cerr << "dspsr: error parsing " << optarg << " as"
	  " number of samples used to estimate undigitized power" << endl;
	return -1;
      }
      break;

    case 'c':
      scanned = sscanf (optarg, "%f", &tbc_cutoff);
      if (scanned != 1) {
        cerr << "dspsr: error parsing " << optarg << " as"
          " dynamic output level assignment cutoff" << endl;
        return -1;
      }
      break;

    case 't':
      scanned = sscanf (optarg, "%f", &tbc_threshold);
      if (scanned != 1) {
        cerr << "dspsr: error parsing " << optarg << " as"
          " sampling threshold" << endl;
        return -1;
      }
      break;

    case 'V':
      dsp::Operation::verbose = true;
    case 'v':
      verbose = true;
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

  if (display) {
    cpgbeg (0, "?", 0, 0);
    cpgask(1);
    cpgsvp (0.05, 0.95, 0.0, 0.8);
    cpgsch (2.0);
  }

  // converted voltages container
  Reference::To<dsp::TimeSeries> voltages = new dsp::TimeSeries;

  // interface manages the creation of data loading and converting classes
  Reference::To<dsp::IOManager> manager = new dsp::IOManager;

  manager->set_block_size (512*tbc_nsample);
  manager->set_output (voltages);

  // plots two-bit digitization statistics
  Reference::To<dsp::TwoBitStatsPlotter> plotter = new dsp::TwoBitStatsPlotter;
  
  Reference::To<dsp::TwoBitCorrection> correct;

  plotter->set_viewport (0.7, 0.95, 0.1, 0.9);
  plotter->horizontal = false;

  for (unsigned ifile=0; ifile < filenames.size(); ifile++) try {

    manager->open (filenames[ifile]);

    if (verbose)
      cerr << "digistat: file " << filenames[ifile] << " opened" << endl;

    // create a new unpacker, appropriate to the backend
    correct = dynamic_cast<dsp::TwoBitCorrection*>(manager->get_unpacker());
    if (!correct) {
      cerr << "digistat: " << filenames[ifile] <<
	" does not contain two-bit data" << endl;
      continue;
    }

    if ( tbc_nsample )
      correct -> set_nsample ( tbc_nsample );

    if ( tbc_threshold )
      correct -> set_threshold ( tbc_threshold );

    if ( tbc_cutoff )
      correct -> set_cutoff_sigma ( tbc_cutoff );

    plotter->set_data (correct);

    while (!manager->eod()) {

      correct->zero_histogram ();
      
      manager->load (voltages);

      if (display)  {
        cpgpage();
        plotter->plot();
      }

      for (unsigned ipol=0; ipol<voltages->get_npol(); ipol++)  {
        float mean = voltages->mean(0, ipol);
        cerr << "mean[" << ipol << "]=" << mean << endl;
      }

    }

    if (verbose)
      cerr << "end of data file " << filenames[ifile] << endl;

  }
  catch (string& error) {
    cerr << error << endl;
  }
  
  if (display)
    cpgend();
  
  return 0;
}


catch (Error& error) {
  cerr << error << endl;
  return -1;
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

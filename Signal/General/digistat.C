#include <iostream>
#include <unistd.h>

#include <cpgplot.h>

#include "dsp/TwoBitStatsPlotter.h"
#include "dsp/TwoBitCorrection.h"
#include "dsp/Chronoseries.h"
#include "dsp/Timeseries.h"
#include "dsp/IOManager.h"
#include "dsp/Input.h"
#include "Error.h"

#include "string_utils.h"
#include "dirutil.h"

static char* args = "vV";

void usage ()
{
  cout << "digistat - plots digitizer statistics\n"
    "Usage: digistat [" << args << "] file1 [file2 ...] \n"
       << endl;
}

int main (int argc, char** argv) 

{ try {

  char* metafile = 0;
  bool display = false;
  bool verbose = false;

  int c;
  while ((c = getopt(argc, argv, args)) != -1)
    switch (c) {

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
  }

  if (display) {
    cpgsvp (0.05, 0.95, 0.0, 0.8);
    cpgsch (2.0);
  }

  // converted voltages container
  Reference::To<dsp::Timeseries> voltages = new dsp::Timeseries;

  // interface manages the creation of data loading and converting classes
  Reference::To<dsp::IOManager> manager = new dsp::IOManager;

  manager->set_block_size (512*512);
  manager->set_final_output (voltages);

  // plots two-bit digitization statistics
  Reference::To<dsp::TwoBitStatsPlotter> plotter = new dsp::TwoBitStatsPlotter;
  
  Reference::To<dsp::TwoBitCorrection> correct;

  plotter->set_viewport (0.7, 0.95, 0.1, 0.9);
  plotter->horizontal = false;

  cpgbeg(0, "?",1,1);
  cpgsch(2);  // set character height

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

    plotter->set_data (correct);

    while (!manager->eod()) {

      correct->zero_histogram ();
      
      manager->load (voltages);

      cpgpage();
      plotter->plot();

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

catch (Reference::invalid& error) {
  cerr << "Invalid Reference exception thrown" << endl;
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

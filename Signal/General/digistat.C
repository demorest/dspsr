#include <iostream>
#include <unistd.h>

#include <cpgplot.h>

#include "TwoBitStatsPlotter.h"
#include "DataManager.h"
#include "TwoBitCorrection.h"
#include "Input.h"
#include "Timeseries.h"

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

  // raw baseband data container
  dsp::Timeseries raw;

  // converted voltages container
  dsp::Timeseries voltage;

  // interface manages the creation of data loading and converting classes
  dsp::DataManager manager;

  // plots two-bit digitization statistics
  dsp::TwoBitStatsPlotter plotter;

  // raw baseband data input
  Reference::To<dsp::Input> input;

  // voltage converter
  Reference::To<dsp::TwoBitCorrection> correct;

  cpgbeg(0, "?",1,1);
  cpgsch(2);  // set character height

  for (unsigned ifile=0; ifile < filenames.size(); ifile++) try {

    manager.open (filenames[ifile]);

    if (verbose)
      cerr << "data file " << filenames[ifile] << " opened" << endl;

    // create a new file input, appropriate to the backend
    input = manager.get_input();

    input->set_block_size (512*512);

    if (verbose)
      cerr << "input initialized" << endl;

    // create a new unpacker, appropriate to the backend
    correct = dynamic_cast<dsp::TwoBitCorrection*>(manager.get_converter());
    if (!correct) {
      cerr << "converter is not a TwoBitCorrection subclass" << endl;
      continue;
    }

    correct->set_input (&raw);
    correct->set_output (&voltage);

    if (verbose)
      cerr << "converter initialized" << endl;

    plotter.set_data (correct);

    while (!input->eod()) {

      correct->zero_histogram ();
      
      input->load (&raw);

      correct->operate ();

      cpgpage();
      plotter.plot();

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

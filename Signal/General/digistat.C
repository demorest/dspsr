#include <iostream>
#include <unistd.h>

#include <cpgplot.h>

#include "TwoBitStatsPlotter.h"
#include "Timeseries.h"
#include "CPSRFileLoader.h"
#include "CPSRTwoBitCorrection.h"
#include "string_utils.h"
#include "dirutil.h"

void usage ()
{
  cout << "program to look at CPSR data using basband/dsp classes\n"
    "Usage: test_CPSRTwoBitCorrection [options] file1 [file2 ...] \n"
       << endl;
}

int main (int argc, char** argv) 
{
  char* metafile = 0;
  bool display = false;
  bool verbose = false;

  int c;
  const char* args = "vV";
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

  dsp::Timeseries raw;
  dsp::Timeseries voltage;

  dsp::CPSRFileLoader loader;
  loader.set_block_size (512*512);
  loader.set_output (&raw);

  dsp::CPSRTwoBitCorrection correct;
  correct.set_input (&raw);
  correct.set_output (&voltage);

  dsp::TwoBitStatsPlotter plotter;
  plotter.set_viewport (0.1, 0.9,  0.1, 0.9);
  plotter.set_data (&correct);

  cpgbeg(0, "?",1,1);
  cpgsch(2);  // set character height


  for (unsigned ifile=0; ifile < filenames.size(); ifile++) try {

    loader.open (filenames[ifile].c_str());

    while (!loader.eod()) {

      correct.zero_histogram ();

      loader.operate ();
      correct.operate ();

      cpgpage();
      plotter.plot();

    }

    cerr << "end of data file " << filenames[ifile] << endl;
    loader.close ();

  }
  catch (string& error) {
    cerr << error << endl;
  }
  
  if (display)
    cpgend();
  
  return 0;
}



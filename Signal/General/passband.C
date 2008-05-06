/***************************************************************************
 *
 *   Copyright (C) 2004 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/Bandpass.h"
#include "dsp/RFIFilter.h"

#include "dsp/IOManager.h"
#include "dsp/MultiFile.h"

#include "dsp/TwoBitCorrection.h"
#include "dsp/WeightedTimeSeries.h"

#include "BandpassPlotter.h"
#include "ColourMap.h"
#include "dirutil.h"
#include "strutil.h"
#include "Error.h"

#include <cpgplot.h>
#include <iostream>
#include <unistd.h>

using namespace std;

void usage ()
{
  cout << "passband - plot passband\n"
    "Usage: passband [options] file1 [file2 ...] \n"
    "Options:\n"
    " -c cmap    set the colour map (0 to 7) \n"
    " -d         produce dynamic spectrum (greyscale) \n"
    " -m maxval  set the maximum value in the greyscale (saturate birdies) \n"
    " -n nchan   number of frequency channels in each spectrum \n"
    " -t seconds integration interval for each spectrum \n"
    " -R         test RFIFilter class \n"
       << endl;
}

int main (int argc, char** argv) try {

  // number of frequency channels in passband
  unsigned nchan = 1024;

  // number of FFTs per block
  unsigned ffts = 16;

  // bandwidth
  double bandwidth = 0.0;

  // centre_frequency
  double centre_frequency = 0.0;

  // integration length
  double integrate = 1.0;

  // seek into file
  double seek_seconds = 0.0;

  // total number of seconds to process
  double total_seconds = 0.0;

  // dynamic spectrum
  bool dynamic = false;

  // load filenames from the ascii file named metafile
  char* metafile = 0;
  bool verbose = false;

  // RFI filter
  dsp::RFIFilter* filter = 0;

  // the plotter
  fft::BandpassPlotter<dsp::Response, dsp::TimeSeries> plotter;

  // the colour map
  pgplot::ColourMap::Name colour_map = pgplot::ColourMap::Heat;
  pgplot::ColourMap cmap;

  // the PGPLOT display
  string display = "?";

  int c;

  static char* args = "B:c:dD:f:lm:n:RS:T:t:hvV";

  while ((c = getopt(argc, argv, args)) != -1)
    switch (c) {

    case 'B':
      bandwidth = atof (optarg);
      break;

    case 'c':
      colour_map = (pgplot::ColourMap::Name) atoi(optarg);
      break;

    case 'D':
      display = optarg;
      break;

    case 'd':
      dynamic = true;
      break;

    case 'f':
      centre_frequency = atof (optarg);
      break;

    case 'g':
      ffts = atoi (optarg);
      break;

    case 'l':
      plotter.logarithmic = true;
      break;

    case 'R':
      filter = new dsp::RFIFilter;
      break;

    case 'S':
      seek_seconds = atof (optarg);
      break;

    case 'T':
      total_seconds = atof (optarg);
      break;

    case 't':
      integrate = atof (optarg);
      break;

    case 'h':
      usage ();
      return 0;

    case 'M':
      metafile = optarg;
      break;
      
    case 'm':
      plotter.user_max = atoi (optarg);
      break;

    case 'n':
      nchan = atoi (optarg);
      break;

    case 'V':
      dsp::Observation::verbose = true;
      dsp::Operation::verbose = true;
      dsp::Shape::verbose = true;
    case 'v':
      verbose = true;
      break;

    default:
      cerr << "invalid param '" << c << "'" << endl;

    }

  vector<string> filenames;

  if (metafile)
    stringfload (&filenames, metafile);
  else 
    for (int ai=optind; ai<argc; ai++)
      dirglob (&filenames, argv[ai]);

  if (filenames.size() == 0) {
    cerr << "passband: please specify a filename" << endl;
    return 0;
  }

  if (verbose)
    cerr << "Creating WeightedTimeSeries instance" << endl;
  dsp::TimeSeries* voltages = new dsp::WeightedTimeSeries;

  if (verbose)
    cerr << "Creating Response (output) instance" << endl;
  dsp::Response* output = new dsp::Response;

  vector<dsp::Operation*> operations;

  if (verbose)
    cerr << "Creating IOManager instance" << endl;

  dsp::IOManager* manager = new dsp::IOManager;
  manager->set_output (voltages);
  operations.push_back (manager);

  if (verbose)
    cerr << "Creating Bandpass instance" << endl;

  dsp::Bandpass* passband = new dsp::Bandpass;
  passband->set_input (voltages);
  passband->set_output (output);
  passband->set_nchan (nchan);
  operations.push_back (passband);

  if (cpgopen(display.c_str()) < 0) {
    cerr << "passband: Could not open plot device" << endl;
    return -1;
  }
  cpgsvp (0.1, 0.9, 0.15, 0.9);
  cmap.set_name (colour_map);

  for (unsigned ifile=0; ifile < filenames.size(); ifile++) try {

    if (verbose)
      cerr << "opening data file " << filenames[ifile] << endl;
      
    manager->open (filenames[ifile]);
    
    if (verbose)
      cerr << "data file " << filenames[ifile] << " opened" << endl;

    if (bandwidth != 0) {
      cerr << "passband: over-riding bandwidth"
              " old=" << manager->get_info()->get_bandwidth() <<
              " new=" << bandwidth << endl;
      manager->get_info()->set_bandwidth (bandwidth);
    }

    if (centre_frequency != 0) {
      cerr << "passband: over-riding centre_frequency"
              " old=" << manager->get_info()->get_centre_frequency() <<
              " new=" << centre_frequency << endl;
      manager->get_info()->set_centre_frequency (centre_frequency);
    }

    if (seek_seconds)
      manager->get_input()->seek_seconds (seek_seconds);

    if (total_seconds)
      manager->get_input()->set_total_seconds (seek_seconds + total_seconds);

    unsigned real_complex = 2 / manager->get_info()->get_ndim();
      
    unsigned block_size = ffts * nchan * real_complex;

    cerr << "Blocksz=" << block_size << endl;

    manager->get_input()->set_block_size ( block_size );
    manager->get_input()->set_overlap ( 0 );

    dsp::TwoBitCorrection* tbc;
    tbc = dynamic_cast<dsp::TwoBitCorrection*> ( manager->get_unpacker() );

    float tbc_cutoff = 100.0;

    if ( tbc && tbc_cutoff )
      tbc -> set_cutoff_sigma ( tbc_cutoff );

#if 0

    if ( tbc && tbc_threshold )
      tbc -> set_threshold ( tbc_threshold );

    if ( tbc && tbc_nsample )
      tbc -> set_nsample ( tbc_nsample );

    if ( tbc && tbc_cutoff )
      tbc -> set_cutoff_sigma ( tbc_cutoff );

#endif

    vector< vector<float> > dynamic_spectrum [2];

    double time_into_file = 0.0;

    while (!manager->get_input()->eod()) {

      for (unsigned iop=0; iop < operations.size(); iop++) try {
	
        if (verbose) cerr << "passband: calling " 
                          << operations[iop]->get_name() << endl;
	
        operations[iop]->operate ();
	
        if (verbose) cerr << "passband: " << operations[iop]->get_name() 
                          << " done" << endl;
	
      }
      catch (Error& error)  {
	
	cerr << "passband: " << operations[iop]->get_name() << " error\n"
	     << error.get_message() << endl;
	
	break;
	
      }

      cerr << ".";

      if (passband->get_integration_length() > integrate) {

	cerr << "\npassband: plotting passband from t=" << time_into_file
	     << "->" << time_into_file + passband->get_integration_length()
	     << " seconds" << endl;

	time_into_file += passband->get_integration_length();

	if (filter)
	  filter->calculate (output);

	if (dynamic)
	  for (unsigned ipol=0; ipol < output->get_npol(); ipol++)
	    dynamic_spectrum[ipol].push_back( output->get_passband(ipol) );
	else {
	  cpgpage ();
	  plotter.plot (output, voltages);
	}

	passband->reset_output();

      }

    }

    if (dynamic)
      for (unsigned ipol=0; ipol < output->get_npol(); ipol++) {
	cerr << "Plotting dynamic spectrum for ipol=" << ipol << endl;
	cpgpage ();
	plotter.plot (dynamic_spectrum[ipol], time_into_file, voltages);
      }

  }
  catch (Error& error)  {
    cerr << "passband: " << filenames[ifile] << " error\n"
	 << error.get_message() << endl;
  }

  if (verbose)
    cerr << "end of data" << endl;
  
  cpgend ();

  return 0;
}

catch (Error& error) {
  cerr << "Error thrown: " << error << endl;
  return -1;
}


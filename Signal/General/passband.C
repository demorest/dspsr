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
#include "dsp/SampleDelay.h"
#include "dsp/GeometricDelay.h"

#include "dsp/ExcisionUnpacker.h"
#include "dsp/WeightedTimeSeries.h"

#include "BandpassPlotter.h"
#include "ColourMap.h"
#include "dirutil.h"
#include "strutil.h"
#include "pgutil.h"
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
    " -b         plot frequency bins (histogram style) \n"
    " -c cmap    set the colour map (0 to 7) \n"
    " -d         produce dynamic spectrum (greyscale) \n"
    " -F min,max set the min,max x-value (e.g. frequency zoom) \n" 
    " -r min,max set the min,max y-value (e.g. saturate birdies) \n"
    " -n nchan   number of frequency channels in each spectrum \n"
    " -t seconds integration interval for each spectrum \n"
    " -p         detect the full-polarization bandpass \n"
    " -R         test RFIFilter class \n"
       << endl;
}

int main (int argc, char** argv) try {

  // number of frequency channels in passband
  unsigned nchan = 1024;

  // number of FFTs per block
  unsigned ffts = 16;

  // detect only the power in each polarization
  Signal::State state = Signal::PPQQ;

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
  plotter.title = "Bandpass";

  // GeometricDelay computes the required delays
  dsp::GeometricDelay* geometry = 0;

  // the colour map
  pgplot::ColourMap::Name colour_map = pgplot::ColourMap::Heat;
  pgplot::ColourMap cmap;

  // the PGPLOT display
  string display = "?";

  int c;

  int width_pixels  = 0;
  int height_pixels = 0;

  static char* args = "ibB:c:dD:f:F:G:g:lr:n:pRS:T:t:hvV";

  while ((c = getopt(argc, argv, args)) != -1)
    switch (c) {

    case 'i':
      plotter.xlabel_ichan = true;
      break;

    case 'b':
      plotter.histogram = true;
      break;

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
    {
      char separator = 0;
      if (sscanf (optarg, "%d%c%d", &width_pixels, &separator, &height_pixels) != 3)
      {
        cerr << "passband: could not parse WxH from '" << optarg << "'" << endl;
        return 0;
      }
      break;
    }

    case 'G':
    {
      geometry = new dsp::GeometricDelay;

      cerr << "passband -G " << optarg << endl;

      /*
	HERE is where I would load the file that describes telescope 
	coordinates and other information, and set GeometricDelay attributes

	The filename is given by 'optarg'
      */

      break;
    }

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
      
    case 'F':
    {
      float min, max;
      sscanf (optarg, "%f,%f", &min, &max);
      plotter.set_fminmax (min, max);
      break;
    }

    case 'r':
    {
      float min, max;
      sscanf (optarg, "%f,%f", &min, &max);
      plotter.set_minmax (min, max);
      break;
    }

    case 'n':
      nchan = atoi (optarg);
      break;

    case 'p':
      state = Signal::Coherence;
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

  if (geometry)
  {
    cerr << "Creating SampleDelay instance" << endl;

    // SampleDelay performs the delay operation
    dsp::SampleDelay* sample_delay = new dsp::SampleDelay;

    // set the delay function 
    sample_delay -> set_function (geometry);
    sample_delay -> set_input (voltages);
    sample_delay -> set_output (voltages);

    operations.push_back (sample_delay);
  }

  if (verbose)
    cerr << "Creating Bandpass instance" << endl;

  dsp::Bandpass* passband = new dsp::Bandpass;
  passband->set_input (voltages);
  passband->set_output (output);
  passband->set_nchan (nchan);
  passband->set_state (state);

  if (geometry)
    passband->set_response( geometry->get_response() );


  operations.push_back (passband);

  if (cpgopen(display.c_str()) < 0)
  {
    cerr << "passband: Could not open plot device" << endl;
    return -1;
  }
  cpgsvp (0.1, 0.9, 0.15, 0.9);
  cmap.set_name (colour_map);

  if (width_pixels && height_pixels)
    pgplot::set_dimensions (width_pixels, height_pixels);

  for (unsigned ifile=0; ifile < filenames.size(); ifile++) try
  {

    if (verbose)
      cerr << "opening data file " << filenames[ifile] << endl;
      
    manager->open (filenames[ifile]);
    
    if (verbose)
      cerr << "data file " << filenames[ifile] << " opened" << endl;

    if (bandwidth != 0)
    {
      cerr << "passband: over-riding bandwidth"
              " old=" << manager->get_info()->get_bandwidth() <<
              " new=" << bandwidth << endl;
      manager->get_info()->set_bandwidth (bandwidth);
    }

    if (centre_frequency != 0)
    {
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

    if (manager->get_info()->get_detected())
    {
      cerr << "passband: input data are detected" << endl;
      block_size = integrate * manager->get_info()->get_rate();
    }

    cerr << "Blocksz=" << block_size << endl;

    manager->set_block_size ( block_size );

    // HERE is where to set overlap
    // manager->set_overlap (nchan * real_complex);

    /*
      HERE the file/ringbuffer is open
      set the delay based on file contents

      sample_delay -> delay_poln (nsamp);
    */

    dsp::ExcisionUnpacker* excision;
    excision = dynamic_cast<dsp::ExcisionUnpacker*>( manager->get_unpacker() );

    if ( excision )
      excision -> set_cutoff_sigma ( 0.0 );

    vector< vector<float> > dynamic_spectrum [2];

    double time_into_file = 0.0;

    while (!manager->get_input()->eod())
    {
      for (unsigned iop=0; iop < operations.size(); iop++) try
      {
        if (verbose) cerr << "passband: calling " 
                          << operations[iop]->get_name() << endl;
	
        operations[iop]->operate ();
	
        if (verbose) cerr << "passband: " << operations[iop]->get_name() 
                          << " done" << endl;
      }
      catch (Error& error)
      {
	cerr << "passband: " << operations[iop]->get_name() << " error\n"
	     << error.get_message() << endl;
	
	break;
      }

      // HERE is where you could write results to disk

      if (passband->get_integration_length() > integrate)
      {
	cerr << "\npassband: plotting passband from t=" << time_into_file
	     << "->" << time_into_file + passband->get_integration_length()
	     << " seconds" << endl;

	time_into_file += passband->get_integration_length();

	if (filter)
	  filter->calculate (output);

	if (dynamic)
	  for (unsigned ipol=0; ipol < output->get_npol(); ipol++)
	    dynamic_spectrum[ipol].push_back( output->get_passband(ipol) );
	else 
	{
	  cpgpage ();
	  output->naturalize ();
	  plotter.plot (output, voltages);
	}

	passband->reset_output();
      }
    }

    if (dynamic)
      for (unsigned ipol=0; ipol < output->get_npol(); ipol++)
      {
	cerr << "Plotting dynamic spectrum for ipol=" << ipol << endl;
	cpgpage ();
	plotter.plot (dynamic_spectrum[ipol], time_into_file, voltages);
      }

  }
  catch (Error& error)
  {
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


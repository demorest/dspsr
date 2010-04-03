/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/*
  digifil converts any file format recognized by dspsr into sigproc
  filterbank (.fil) format.
 */

#include "dsp/SigProcObservation.h"
#include "dsp/SigProcDigitizer.h"

#include "dsp/IOManager.h"
#include "dsp/Input.h"
#include "dsp/Unpacker.h"

#include "dsp/Rescale.h"
#include "dsp/PScrunch.h"
#include "dsp/TScrunch.h"
#include "dsp/Filterbank.h"
#include "dsp/Detection.h"

#include "dirutil.h"
#include "Error.h"

#include <iostream>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

static char* args = "b:B:F:cI:o:prt:hvVZ:";

void usage ()
{
  cout << "digifil - convert dspsr input to sigproc filterbank \n"
    "Usage: digifil file1 [file2 ...] \n"
    "Options:\n"
    "\n"
    "  -b bits   number of bits per sample output to file \n" 
    "  -B MB     block size in megabytes \n"
    "  -c        keep offset and scale constant \n"
    "  -I secs   number of seconds between level updates \n"
    "  -F nchan  create a filterbank (voltages only) \n"
    "  -t nsamp  decimate in time \n"
    "  -I secs   rescale interval in seconds \n"
    "  -o file   output filename \n" 
    "  -r        report total Operation times \n"
    "  -p        revert to FPT order \n"
       << endl;
}

int main (int argc, char** argv) try 
{
  bool verbose = false;
  bool constant_offset_scale = false;

  int nbits = 2;
  unsigned filterbank_nchan = 0;
  unsigned tscrunch_factor = 0;

  // block size in MB
  double block_size = 2.0;
  double rescale_seconds = 10.0;

  char* output_filename = 0;

  dsp::TimeSeries::Order order = dsp::TimeSeries::OrderTFP;

  int c;
  while ((c = getopt(argc, argv, args)) != -1)
    switch (c) {

    case 'b':
      nbits = atoi (optarg);
      break;

    case 'B':
      block_size = atof (optarg);
      break;

    case 'c':
      constant_offset_scale = true;
      break;

    case 'F':
      filterbank_nchan = atoi (optarg);
      break;

    case 'o':
      output_filename = optarg;
      break;

    case 'p':
      order = dsp::TimeSeries::OrderFPT;
      break;

    case 'r':
      dsp::Operation::record_time = true;
      dsp::Operation::report_time = true;
      break;

    case 't':
      tscrunch_factor = atoi (optarg);
      break;

    case 'I':
      rescale_seconds = atof (optarg);
      break;

    case 'h':
      usage ();
      return 0;
    case 'V':
      dsp::Operation::verbose = true;
      dsp::Observation::verbose = true;
    case 'v':
      verbose = true;
      break;

    case 'Z':
    {
      string lib = optarg;

      if (lib == "help")
      {
        unsigned nlib = FTransform::get_num_libraries ();
        cerr << "dspsr: " << nlib << " available FFT libraries:";
        for (unsigned ilib=0; ilib < nlib; ilib++)
          cerr << " " << FTransform::get_library_name (ilib);
  
        cerr << "\ndspsr: default FFT library " 
             << FTransform::get_library() << endl;

        exit (0);
      }
      else if (lib == "simd")
        FTransform::simd = true;
      else {
        FTransform::set_library (lib);
        cerr << "dspsr: FFT library set to " << lib << endl;
      }

      break;
    }
    default:
      cerr << "invalid param '" << c << "'" << endl;
    }

  vector <string> filenames;
  
  for (int ai=optind; ai<argc; ai++)
    dirglob (&filenames, argv[ai]);

  if (filenames.size() == 0)
  {
    cerr << "digifil: please specify a filename (-h for help)" 
	 << endl;
    return 0;
  }

  FILE* outfile = stdout;

  if (!output_filename)
    fprintf (stderr, "digifil: output on stdout \n");
  else
  {
    outfile = fopen (output_filename, "w");
    if (!outfile)
    {
      fprintf (stderr, "digifil: could not open %s - %s\n",
	       output_filename, strerror (errno));
      return -1;
    }
  }

  /*
    The following lines "wire up" the signal path, using containers
    to communicate the data between operations.
  */

  if (verbose)
    cerr << "digifil: creating input timeseries container" << endl;
  Reference::To<dsp::TimeSeries> timeseries = new dsp::TimeSeries;

  if (verbose)
    cerr << "digifil: creating input/unpacker manager" << endl;
  Reference::To<dsp::IOManager> manager = new dsp::IOManager;
  manager->set_output (timeseries);

  Reference::To<dsp::Rescale> rescale;
  if (rescale_seconds)
  {
    if (verbose)
      cerr << "digifil: creating rescale transformation" << endl;
    rescale = new dsp::Rescale;
    rescale->set_input (timeseries);
    rescale->set_output (timeseries);
    rescale->set_constant (constant_offset_scale);
    rescale->set_interval_seconds (rescale_seconds);
  }

  if (verbose)
    cerr << "digifil: creating pscrunch transformation" << endl;
  Reference::To<dsp::PScrunch> pscrunch = new dsp::PScrunch;
  pscrunch->set_input (timeseries);
  pscrunch->set_output (timeseries);

  if (verbose)
    cerr << "digifil: creating output bitseries container" << endl;
  Reference::To<dsp::BitSeries> bitseries = new dsp::BitSeries;

  if (verbose)
    cerr << "digifil: creating sigproc digitizer" << endl;
  Reference::To<dsp::SigProcDigitizer> digitizer = new dsp::SigProcDigitizer;
  digitizer->set_nbit (nbits);
  digitizer->set_input (timeseries);
  digitizer->set_output (bitseries);

  bool written_header = false;

  for (unsigned ifile=0; ifile < filenames.size(); ifile++) try
  {
    Reference::To<dsp::Filterbank> filterbank;
    Reference::To<dsp::TimeSeries> filterbank_input;
    Reference::To<dsp::Detection> detection;
    Reference::To<dsp::TScrunch> tscrunch;

    if (verbose)
      cerr << "digifil: opening file " << filenames[ifile] << endl;

    manager->open (filenames[ifile]);

    dsp::Observation* obs = manager->get_info();
    const unsigned nchan = obs->get_nchan ();
    const unsigned npol = obs->get_npol ();
    const unsigned ndim = obs->get_ndim ();

    // the unpacked input will occupy nbytes_per_sample
    double nbytes_per_sample = sizeof(float) * nchan * npol * ndim;

    double MB = 1024.0 * 1024.0;
    uint64_t nsample = uint64_t( block_size*MB / nbytes_per_sample );

    if (verbose)
      cerr << "digifil: block_size=" << block_size << " MB "
        "(" << nsample << " samp)" << endl;

    bool do_pscrunch = obs->get_npol() > 1;

    if (!obs->get_detected())
    {

      if (filterbank_nchan)
      {
	if (verbose)
	  cerr << "digifil: creating " << filterbank_nchan 
	       << " channel filterbank" << endl;

	filterbank = new dsp::Filterbank;
	filterbank_input = new dsp::TimeSeries;

	manager->set_output (filterbank_input);

	filterbank->set_nchan( filterbank_nchan );
	filterbank->set_input( filterbank_input );
	filterbank->set_output( timeseries );

        // filterbank will do detection
        filterbank->set_output_order( dsp::TimeSeries::OrderTFP );
      }
      else
      {
	if (verbose)
	  cerr << "digifil: detecting data directly" << endl;

        detection = new dsp::Detection;
        detection->set_input( timeseries );
        detection->set_output( timeseries );

        // detection will do pscrunch
        do_pscrunch = false;
      }
    }

    if (tscrunch_factor)
    {
      tscrunch = new dsp::TScrunch;

      tscrunch->set_factor (tscrunch_factor);
      tscrunch->set_input( timeseries );
      tscrunch->set_output( timeseries );
    }

    manager->set_block_size( nsample );

    dsp::Unpacker* unpacker = manager->get_unpacker();

    if (unpacker->get_order_supported (order))
      unpacker->set_output_order (order);

    if (verbose)
    {
      dsp::Observation* obs = manager->get_info();

      cerr << "digifil: file " 
	   << filenames[ifile] << " opened" << endl;
      cerr << "Source = " << obs->get_source() << endl;
      cerr << "Frequency = " << obs->get_centre_frequency() << endl;
      cerr << "Bandwidth = " << obs->get_bandwidth() << endl;
      cerr << "Sampling rate = " << obs->get_rate() << endl;
    }

    dsp::SigProcObservation sigproc;

    while (!manager->get_input()->eod())
    {
      manager->operate ();

      if (filterbank)
      {
	if (verbose)
	  cerr << "digifil: filterbank" << endl;

	filterbank->operate();
      }

      if (detection)
      {
	if (verbose)
	  cerr << "digifil: detection" << endl;

	detection->operate();
      }

      if (tscrunch)
      {
	if (verbose)
	  cerr << "digifil: tscrunch" << endl;

        tscrunch->operate();
      }

      if (rescale)
      {
	if (verbose)
	  cerr << "digifil: rescale" << endl;
      
	rescale->operate ();
      }

      if (do_pscrunch)
      {
	if (verbose)
	  cerr << "digifil: tscrunch" << endl;
	  
	pscrunch->operate ();
      }

      digitizer->operate ();

      if (!written_header)
      {
	if (verbose)
	  cerr << "digifil: unload header" << endl;

	sigproc.copy( bitseries );
	sigproc.unload( outfile );
	written_header = true;
      }

      // output the result to stdout
      const uint64_t nbyte = bitseries->get_nbytes();
      unsigned char* data = bitseries->get_rawptr();

      if (verbose)
	cerr << "digifil: writing " << nbyte << " bytes to file" << endl;

      fwrite (data,nbyte,1,outfile);

    }

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





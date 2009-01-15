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

static char* args = "b:B:F:co:prT:hvV";

void usage ()
{
  cout << "digifil - convert dspsr input to sigproc filterbank \n"
    "Usage: digifil file1 [file2 ...] \n"
    "Options:\n"
    "\n"
    "  -b bits   number of bits per sample output to file \n" 
    "  -B secs   block size in seconds \n"
    "  -c        keep offset and scale constant \n"
    "  -F nchan  create a filterbank (voltages only) \n"
    "  -T nsamp  decimate in time \n"
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

  unsigned nbits = 2;
  unsigned filterbank_nchan = 0;
  unsigned tscrunch_factor = 0;

  // block size in seconds
  double block_size = 10;

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
      break;

    case 'T':
      tscrunch_factor = atoi (optarg);
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

  if (output_filename)
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

  if (verbose)
    cerr << "digifil: creating rescale transformation" << endl;
  Reference::To<dsp::Rescale> rescale = new dsp::Rescale;
  rescale->set_input (timeseries);
  rescale->set_output (timeseries);
  rescale->set_constant (constant_offset_scale);

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
  digitizer->set_nbit(nbits);
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

    uint64 nsample = uint64( block_size * obs->get_rate() );
    bool do_pscrunch = obs->get_npol() > 1;

    if (verbose)
      cerr << "digifil: block_size=" << block_size << " sec "
        "(" << nsample << " samp)" << endl;

    if (!obs->get_detected())
    {

      if (filterbank_nchan)
      {
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
        detection = new dsp::Detection;
        detection->set_input( timeseries );
        detection->set_output( timeseries );

        // detection will do pscrunch
        do_pscrunch = false;
      }
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

    if (tscrunch_factor)
    {
      tscrunch = new dsp::TScrunch;

      tscrunch->set_factor (tscrunch_factor);
      tscrunch->set_input( timeseries );
      tscrunch->set_output( timeseries );
    }

    dsp::SigProcObservation sigproc;

    while (!manager->get_input()->eod())
    {
      manager->operate ();

      if (filterbank)
	filterbank->operate();

      if (detection)
	detection->operate();

      if (tscrunch)
        tscrunch->operate();

      rescale->operate ();

      if (do_pscrunch)
	pscrunch->operate ();

      digitizer->operate ();

      if (!written_header)
      {
	sigproc.copy( bitseries );
	sigproc.unload( outfile );
	written_header = true;
      }

      // output the result to stdout
      const uint64 nbyte = bitseries->get_nbytes();
      unsigned char* data = bitseries->get_rawptr();

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





/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SigProcObservation.h"
#include "dsp/SigProcDigitizer.h"

#include "dsp/IOManager.h"
#include "dsp/Input.h"

#include "dsp/Rescale.h"
#include "dsp/PScrunch.h"

#include "dirutil.h"
#include "Error.h"

#include <iostream>
#include <unistd.h>

using namespace std;

static char* args = "hvV";

void usage ()
{
  cout << "sigproc_filterbank - convert dspsr input to sigproc header \n"
    "Usage: sigproc_filterbank file1 [file2 ...] \n" << endl;
}

int main (int argc, char** argv) try 
{
  bool verbose = false;

  // a mega-sample at a time
  uint64 block_size = 1024 * 1024;

  int c;
  while ((c = getopt(argc, argv, args)) != -1)
    switch (c) {

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
    cerr << "sigproc_filterbank: please specify a filename (-h for help)" 
	 << endl;
    return 0;
  }

  /*
    The following lines "wire up" the signal path, using containers
    to communicate the data between operations.
  */

  if (verbose)
    cerr << "sigproc_filterbank: creating input timeseries container" << endl;
  Reference::To<dsp::TimeSeries> timeseries = new dsp::TimeSeries;

  if (verbose)
    cerr << "sigproc_filterbank: creating input/unpacker manager" << endl;
  Reference::To<dsp::IOManager> manager = new dsp::IOManager;
  manager->set_output (timeseries);

  if (verbose)
    cerr << "sigproc_filterbank: creating rescale transformation" << endl;
  Reference::To<dsp::Rescale> rescale = new dsp::Rescale;
  rescale->set_input (timeseries);
  rescale->set_output (timeseries);

  if (verbose)
    cerr << "sigproc_filterbank: creating pscrunch transformation" << endl;
  Reference::To<dsp::PScrunch> pscrunch = new dsp::PScrunch;
  pscrunch->set_input (timeseries);
  pscrunch->set_output (timeseries);

  if (verbose)
    cerr << "sigproc_filterbank: creating output bitseries container" << endl;
  Reference::To<dsp::BitSeries> bitseries = new dsp::BitSeries;

  if (verbose)
    cerr << "sigproc_filterbank: creating sigproc digitizer" << endl;
  Reference::To<dsp::SigProcDigitizer> digitizer = new dsp::SigProcDigitizer;
  digitizer->set_input (timeseries);
  digitizer->set_output (bitseries);

  for (unsigned ifile=0; ifile < filenames.size(); ifile++) try
  {
    if (verbose)
      cerr << "sigproc_filterbank: opening file " << filenames[ifile] << endl;

    manager->open (filenames[ifile]);

    unsigned nchan = manager->get_info()->get_nchan();

    manager->get_input()->set_block_size( block_size/nchan );

    if (verbose)
    {
      dsp::Observation* obs = manager->get_info();

      cerr << "sigproc_filterbank: file " 
	   << filenames[ifile] << " opened" << endl;
      cerr << "Source = " << obs->get_source() << endl;
      cerr << "Frequency = " << obs->get_centre_frequency() << endl;
      cerr << "Bandwidth = " << obs->get_bandwidth() << endl;
      cerr << "Sampling rate = " << obs->get_rate() << endl;
    }

    dsp::SigProcObservation sigproc;

    sigproc.copy( manager->get_info() );
    sigproc.unload( stdout );

    bool do_pscrunch = manager->get_info()->get_npol() > 1;

    while (!manager->get_input()->eod())
    {
      manager->operate ();
      rescale->operate ();

      if (do_pscrunch)
	pscrunch->operate ();

      digitizer->operate ();

      // output the result to stdout
      const uint64 nbyte = bitseries->get_nbytes();
      unsigned char* data = bitseries->get_rawptr();

      for (uint64 ibyte=0; ibyte<nbyte; ibyte++)
	cout << data[ibyte];
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

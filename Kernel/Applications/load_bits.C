/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/BitSeries.h"
#include "dsp/File.h"
#include "Error.h"

#include "strutil.h"
#include "dirutil.h"
#include "templates.h"

#include <iostream>
#include <unistd.h>

using namespace std;

static char* args = "hvV";

void usage ()
{
  cout << "load_bits - load bits from file and print them to stdout\n"
    "Usage: load_bits file1 [file2 ...] \n" << endl;
}

int main (int argc, char** argv) try 
{
  bool verbose = false;

  // number of time samples loaded at once
  uint64_t block_size = 1024;

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
    cerr << "load_bits: please specify a filename (or -h for help)" << endl;
    return 0;
  }

  // generalized interface to input data
  Reference::To<dsp::Input> input;

  // container into which bits are loaded
  Reference::To<dsp::BitSeries> bits;

  for (unsigned ifile=0; ifile < filenames.size(); ifile++) try
  {
    if (verbose)
      cerr << "load_bits: opening file " << filenames[ifile] << endl;

    input = dsp::File::create( filenames[ifile] );

    if (verbose)
    {
      cerr << "load_bits: file " << filenames[ifile] << " opened" << endl;
      cerr << "Bandwidth = " << input->get_info()->get_bandwidth() << endl;
      cerr << "Sampling rate = " << input->get_info()->get_rate() << endl;
    }

    bits = new dsp::BitSeries;

    input->set_block_size( block_size );

    while (!input->eod())
    {
      input->load (bits);

      const unsigned nchan = bits->get_nchan();
      const unsigned npol = bits->get_npol();
      const unsigned ndim = bits->get_ndim();
      const unsigned ndat = bits->get_ndat();

      const unsigned nbyte = bits->get_nbytes();

      if (verbose)
        cerr << "load_bits: nchan=" << nchan << " npol=" << npol 
	     << " ndim=" << ndim << "ndat=" << ndat << " nbyte=" << nbyte
	     << " offset=" << bits->get_input_sample () << endl;

      unsigned char* data = bits->get_datptr (0);

      unsigned out = 0;
      for (unsigned ibyte=0; ibyte < nbyte; ibyte++)
      {
	fprintf (stdout, "%02x ", data[ibyte]);
	out ++;
	if (out == 16)
	{
	  cout << endl;
	  out = 0;
	}
      }
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

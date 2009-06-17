/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/UnpackerIterator.h"
#include "dsp/ExcisionUnpacker.h"
#include "dsp/IOManager.h"

#include "dsp/BitSeries.h"
#include "dsp/Input.h"

#include "Error.h"

#include "strutil.h"
#include "dirutil.h"
#include "templates.h"

#include <iostream>
#include <unistd.h>

using namespace std;

static char* args = "hvVP:";

void usage ()
{
  cout << "digihist - load bits from file and print them to stdout\n"
    "Usage: digihist file1 [file2 ...] \n" << endl;
}

void summarize (vector< vector<uint64_t> >& histogram,
		const dsp::ExcisionUnpacker*,
		unsigned nbit, double start, double end);

int main (int argc, char** argv) try 
{
  bool verbose = false;

  // number of time samples loaded at once
  uint64_t block_size = 1024;

  // period (in seconds) of elapsed recording time between updates
  double update_period = 1.0;

  int c;
  while ((c = getopt(argc, argv, args)) != -1)
    switch (c) {

    case 'h':
      usage ();
      return 0;

    case 'P':
      update_period = atof (optarg);
      break;

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
    cerr << "digihist: please specify a filename (or -h for help)" << endl;
    return 0;
  }

  // interface manages the creation of data loading and converting classes
  Reference::To<dsp::IOManager> manager = new dsp::IOManager;

  // container into which bits are loaded
  Reference::To<dsp::BitSeries> bits = new dsp::BitSeries;

  manager->set_output (bits);

  for (unsigned ifile=0; ifile < filenames.size(); ifile++) try
  {
    if (verbose)
      cerr << "digihist: opening file " << filenames[ifile] << endl;

    manager->open( filenames[ifile] );
    manager->set_block_size( block_size );

    const dsp::Observation* info = manager->get_info();

    if (verbose)
    {
      cerr << "digihist: file " << filenames[ifile] << " opened" << endl;
      cerr << "Bandwidth = " << info->get_bandwidth() << endl;
      cerr << "Sampling rate = " << info->get_rate() << endl;
    }

    dsp::Unpacker* unpacker = manager->get_unpacker();
    dsp::ExcisionUnpacker* excision = 0;
    excision = dynamic_cast<dsp::ExcisionUnpacker*>( unpacker );

    MJD start = info->get_start_time();
    double next_update = update_period;

    vector< vector<uint64_t> > histograms;

    unsigned nbit = info->get_nbit();


    while (!manager->get_input()->eod())
    {
      manager->operate ();

      unsigned char* data = bits->get_datptr ();
      unsigned char* end_of_data = data + bits->get_nbytes ();

      unsigned ndig = 1;
      if (excision)
        ndig = excision->get_ndig ();

      if (histograms.size() < ndig)
      {
        vector<uint64_t> hist_init (256, 0);
        histograms.resize( ndig );
        for (unsigned i=0; i<ndig; i++)
          histograms[i] = hist_init;
      }

      for (unsigned idig=0; idig<ndig; idig++)
      {
	dsp::Unpacker::Iterator iterator = unpacker->get_iterator (idig);

	while (iterator < end_of_data)
	{
	  histograms[idig][ *iterator ] ++;
	  ++ iterator;
	}
      }

      if (next_update &&
	  (bits->get_start_time() - start).in_seconds() >= next_update)
      {
	summarize (histograms, excision, nbit, 
		   next_update-update_period, next_update);

	next_update += update_period;
      }
    }

    if (verbose)
      cerr << "end of data file " << filenames[ifile] << endl;

    if (next_update == 0.0)
      summarize (histograms, excision, nbit,
		 0, (bits->get_end_time() - start).in_seconds());
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


void rebin (vector<uint64_t>& output, 
	    const vector<uint64_t>& input, unsigned nbit)
{
  unsigned nsamp = 8 / nbit;

  // can handle only 2 or 4 bit
  assert (nbit == 2 || nbit == 4);
  unsigned mask = (nbit == 2) ? 0x03 : 0x0f;

  // can handle only 256 input states
  assert (input.size() == 256);

  for (unsigned i=0; i < input.size(); i++)
  {
    unsigned byte = i;
    unsigned count = input[i];

    for (unsigned isamp = 0; isamp < nsamp; isamp++)
    {
      unsigned sample = byte & mask;
      assert (sample < output.size());
      output[sample] += count;
      byte >>= nbit;
    }
  }
}

template<typename T>
void zero (vector<T>& data)
{
  for (unsigned i=0; i<data.size(); i++)
    data[i] = 0;
}

void summarize (const vector<uint64_t>& histogram,
		unsigned nbit, 
		double start, double end,
		unsigned ichan, unsigned ipol)
{
  vector<uint64_t> result;

  switch (nbit)
    {
    case 8:
      result = histogram;
      break;

    case 4:
      result.resize (16);
      zero (result);
      rebin (result, histogram, 4);
      break;

    case 2:
      result.resize (4);
      zero (result);
      rebin (result, histogram, 2);
      break;

    default:
      throw Error (InvalidState, "summarize", "nbit=%d not implemented", nbit);
    }

  uint64_t total = 0;
  for (unsigned i=0; i<result.size(); i++)
    total += result[i];

  for (unsigned i=0; i<result.size(); i++)
    cout << start << "->" << end << " " << ichan << " " << ipol
	 << " " << i << " " << double(result[i])/double(total) << endl;
}

void summarize (vector< vector<uint64_t> >& histograms,
		const dsp::ExcisionUnpacker* excision,
		unsigned nbit, double start, double end)
{
  unsigned ndig = 1;
  
  if (excision)
    ndig = excision->get_ndig ();

  for (unsigned idig=0; idig<ndig; idig++)
  {
    unsigned ipol = 0;
    unsigned ichan = 0;

    if (excision)
    {
      ipol = excision->get_output_ipol (idig);
      ichan = excision->get_output_ichan (idig);
    }

    cerr << "idig=" << idig << " ichan=" << ichan << " ipol=" << ipol << endl;

    summarize (histograms[idig], nbit,
	       start, end, ichan, ipol);

    zero (histograms[idig]);
  }
}

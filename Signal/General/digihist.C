/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/BitSeries.h"
#include "dsp/File.h"
#include "dsp/ExcisionUnpacker.h"

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

void summarize (vector< vector<uint64> >& histogram,
		const dsp::ExcisionUnpacker*,
		unsigned nbit, double start, double end);

int main (int argc, char** argv) try 
{
  bool verbose = false;

  // number of time samples loaded at once
  uint64 block_size = 1024;

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

  // generalized interface to input data
  Reference::To<dsp::Input> input;

  // generalized interface to unpacker
  Reference::To<dsp::Unpacker> unpacker;

  // container into which bits are loaded
  Reference::To<dsp::BitSeries> bits;

  for (unsigned ifile=0; ifile < filenames.size(); ifile++) try
  {
    if (verbose)
      cerr << "digihist: opening file " << filenames[ifile] << endl;

    input = dsp::File::create( filenames[ifile] );

    if (verbose)
    {
      cerr << "digihist: file " << filenames[ifile] << " opened" << endl;
      cerr << "Bandwidth = " << input->get_info()->get_bandwidth() << endl;
      cerr << "Sampling rate = " << input->get_info()->get_rate() << endl;
    }

    const dsp::Observation* info = input->get_info();

    unpacker = dsp::Unpacker::create( input->get_info() );

    dsp::ExcisionUnpacker* excision = 0;
    unsigned ndig = 1;

    if (unpacker)
    {
      excision = dynamic_cast<dsp::ExcisionUnpacker*>( unpacker.ptr() );
      ndig = excision->get_ndig ();
    }

    MJD start = info->get_start_time();
    double next_update = update_period;

    vector<uint64> hist_init (256, 0);
    vector< vector<uint64> > histograms ( ndig, hist_init );

    unsigned nbit = info->get_nbit();

    bits = new dsp::BitSeries;
    input->set_block_size( block_size );

    while (!input->eod())
    {
      input->load (bits);

      const unsigned nbyte = bits->get_nbytes ();
      unsigned char* data = bits->get_datptr ();

      for (unsigned idig=0; idig<ndig; idig++)
      {
	unsigned ipol = 0;
	unsigned ichan = 0;

	unsigned input_offset = 0;
	unsigned input_incr = 1;

	if (excision)
	{
	  ipol = excision->get_output_ipol (idig);
	  ichan = excision->get_output_ichan (idig);

	  input_offset = excision->get_input_offset (idig);
	  input_incr = excision->get_input_incr ();
	}

	for (unsigned ibyte=input_offset; ibyte < nbyte; ibyte+=input_incr)
	  histograms[idig][ data[ibyte] ] ++;
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


void rebin (vector<uint64>& output, 
	    const vector<uint64>& input, unsigned nbit)
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

void summarize (const vector<uint64>& histogram,
		unsigned nbit, 
		double start, double end,
		unsigned ichan, unsigned ipol)
{
  vector<uint64> result;

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

  uint64 total = 0;
  for (unsigned i=0; i<result.size(); i++)
    total += result[i];

  for (unsigned i=0; i<result.size(); i++)
    cout << start << "->" << end << " " << ichan << " " << ipol
	 << " " << i << " " << double(result[i])/double(total) << endl;
}

void summarize (vector< vector<uint64> >& histograms,
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

    summarize (histograms[idig], nbit,
	       start, end, ichan, ipol);

    zero (histograms[idig]);
  }
}

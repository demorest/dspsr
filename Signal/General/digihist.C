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
  cout << "digihist - load bits from file and print them to stdout\n"
    "Usage: digihist file1 [file2 ...] \n" << endl;
}

void summarize (vector<unsigned long>& histogram, unsigned nbit);

int main (int argc, char** argv) try 
{
  bool verbose = false;

  // number of time samples loaded at once
  uint64 block_size = 1024;

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
    cerr << "digihist: please specify a filename (or -h for help)" << endl;
    return 0;
  }

  // generalized interface to input data
  Reference::To<dsp::Input> input;

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

    bits = new dsp::BitSeries;

    input->set_block_size( block_size );

    while (!input->eod())
    {
      input->load (bits);

      unsigned nbit = bits->get_nbit();
      const unsigned nbyte = bits->get_nbytes();
      unsigned char* data = bits->get_datptr (0);

      vector<unsigned long> histogram (256, 0);

      for (unsigned ibyte=0; ibyte < nbyte; ibyte++)
	histogram[ data[ibyte] ] ++;

      summarize (histogram, nbit);
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


template<typename T>
void zero (vector<T>& data)
{
  for (unsigned i=0; i<data.size(); i++)
    data[i] = 0.0;
}

void rebin (vector<unsigned long>& output, 
	    const vector<unsigned long>& input, unsigned nbit)
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

void summarize (vector<unsigned long>& histogram, unsigned nbit)
{
  vector<unsigned long> result;

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

  unsigned long total = 0;
  for (unsigned i=0; i<result.size(); i++)
    total += result[i];

  for (unsigned i=0; i<result.size(); i++)
    cout << i << " " << double(result[i])/double(total) << endl;
}

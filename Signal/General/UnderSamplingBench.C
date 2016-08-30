/***************************************************************************
 *
 *   Copyright (C) 2015 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/UnderSamplingBench.h"
#include "debug.h"

#include <fstream>
#include <math.h>

using namespace std;

bool dsp::UnderSamplingBench::verbose = false;

dsp::UnderSamplingBench::UnderSamplingBench (const std::string& name)
{
  library = name;
  nchan = 0;
}

//! Set the number of channels
void dsp::UnderSamplingBench::set_nchan (unsigned _chan)
{
  if (_chan != nchan)
    reset ();

  nchan = _chan;
}

void dsp::UnderSamplingBench::load () const
{
  max_nfft = 0;

  string filename = path + "/filterbank_bench_" + library + ".dat";

  if (verbose)
    cerr << "dsp::UnderSamplingBench::load filename=" << filename << endl;

  load (library, filename);
  loaded = true;
}

void dsp::UnderSamplingBench::load (const std::string& library,
			      const std::string& filename) const
{
  ifstream in (filename.c_str());
  if (!in)
    throw Error (FailedSys, "dsp::UnderSamplingBench::load",
                 "std::ifstream (" + filename + ")");

  while (!in.eof())
  {
    Entry entry;
    double log2nchan, log2nfft, mflops;
    unsigned _chan;

    in >> _chan >> entry.nfft >> entry.cost >> log2nchan >> log2nfft >> mflops;

    if (in.eof())
      continue;

    entry.library = library;
    
    DEBUG(library << " " << _chan << " " << entry.nfft << " " << entry.cost);

    if (_chan != nchan)
      continue;

    DEBUG("ADD nchan=" << nchan << " nfft=" << entry.nfft);
    entries.push_back (entry);

    if (entry.nfft > max_nfft)
      max_nfft = entry.nfft;
  }
}


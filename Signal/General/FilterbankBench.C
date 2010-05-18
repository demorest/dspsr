/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/FilterbankBench.h"
#include "debug.h"

#include <fstream>
#include <math.h>

using namespace std;

bool dsp::FilterbankBench::verbose = false;

dsp::FilterbankBench::FilterbankBench (const std::string& name)
{
  library = name;
  nchan = 0;
}

//! Set the number of channels
void dsp::FilterbankBench::set_nchan (unsigned _chan)
{
  if (_chan != nchan)
    reset ();

  nchan = _chan;
}

void dsp::FilterbankBench::load () const
{
  max_nfft = 0;

  string filename = path + "/filterbank_bench_" + library + ".dat";

  if (verbose)
    cerr << "dsp::FilterbankBench::load filename=" << filename << endl;

  load (library, filename);
  loaded = true;
}

void dsp::FilterbankBench::load (const std::string& library,
			      const std::string& filename) const
{
  ifstream in (filename.c_str());
  if (!in)
    throw Error (FailedSys, "dsp::FilterbankBench::load",
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


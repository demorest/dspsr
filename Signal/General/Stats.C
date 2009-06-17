/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Stats.h"

#include <fstream>
using namespace std;

dsp::Stats::Stats (const char* name) : Sink<TimeSeries> (name) { }
    
dsp::Stats::~Stats ()
{
  ofstream out ("pdmp.stats");
  if (!out)
    throw Error (FailedSys, "dsp::Stats::~Stats",
		 "could not open pdmp.stats");

  const unsigned nchan = input->get_nchan();
  const unsigned npol = input->get_npol();

  out << "# nchan " << nchan << endl;
  out << "# ntot " << total << endl;

  for (unsigned ichan=0; ichan < nchan; ichan++)
    out << get_sigma(ichan,0) << endl;
}

//! Returns mean in given chan,pol
float dsp::Stats::get_mean (unsigned ichan, unsigned ipol)
{
  return sum[ichan][ipol] / total;
}

template<typename T> T sqr (T x) { return x*x; }
    
//! Returns standard deviation in given chan,pol
float dsp::Stats::get_sigma (unsigned ichan, unsigned ipol)
{
  return sqrt( sumsq[ichan][ipol] / total - sqr(get_mean(ichan,ipol)) );
}

//! Adds to the totals
void dsp::Stats::calculation ()
{
  const unsigned nchan = input->get_nchan();
  const unsigned npol = input->get_npol();
  const uint64_t ndat = input->get_ndat();

  if (sum.size() < nchan)
    init ();

  for (unsigned ichan = 0; ichan < nchan; ichan ++)
    for (unsigned ipol = 0; ipol < npol; ipol++)
    {
      const float* data = input->get_datptr (ichan, ipol);
      for (uint64_t idat = 0; idat < ndat; idat++)
      {
	sum[ichan][ipol] += data[idat];
	sumsq[ichan][ipol] += data[idat] * data[idat];
      }
    }

  total += ndat;
}

void dsp::Stats::init ()
{
  const unsigned nchan = input->get_nchan();
  const unsigned npol = input->get_npol();

  sum.resize (nchan);
  sumsq.resize (nchan);

  for (unsigned ichan=0; ichan < nchan; ichan++)
  {
    sum[ichan].resize (npol);
    sumsq[ichan].resize (npol);

    for (unsigned ipol=0; ipol < npol; ipol++)
    {
      sum[ichan][ipol] = 0;
      sumsq[ichan][ipol] = 0;
    }
  }

  total = 0;
}

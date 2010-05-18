/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/OptimalFilterbank.h"
#include "Pulsar/Config.h"

#include <math.h>

using namespace std;

bool dsp::OptimalFilterbank::verbose = false;

dsp::OptimalFilterbank::OptimalFilterbank (const std::string& _library)
{
  nchan = 1;
  library = _library;

  DEBUG("dsp::OptimalFilterbank library=" << library);
}

void dsp::OptimalFilterbank::set_nchan (unsigned n)
{
  nchan = n;

  if (!fbench)
    bench = new_bench ();
  else
    fbench->set_nchan( nchan );
}

FTransform::Bench* dsp::OptimalFilterbank::new_bench () const
{
  DEBUG("dsp::OptimalFilterbank::new_bench nchan=" << nchan);

  fbench = new FilterbankBench (library);
  fbench->set_path( Pulsar::Config::get_runtime() );
  fbench->set_nchan( nchan );

  return fbench;
}

std::string dsp::OptimalFilterbank::get_library (unsigned nfft)
{
  return library;
}


double dsp::OptimalFilterbank::compute_cost (unsigned nfft, unsigned nfilt) const
{
  DEBUG("dsp::OptimalFilterbank::compute_cost nfft=" << nfft << " nfilt=" << nfilt);
  return fbench->get_best( nfft ).cost / ((nfft - nfilt) * nchan);
}


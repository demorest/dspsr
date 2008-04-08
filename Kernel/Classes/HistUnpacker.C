/***************************************************************************
 *
 *   Copyright (C) 2005 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
// #define _DEBUG 1

#include "dsp/HistUnpacker.h"

#include <iostream>

using namespace std;

bool dsp::HistUnpacker::keep_histogram = true;

//! Null constructor
dsp::HistUnpacker::HistUnpacker (const char* _name) : Unpacker (_name)
{
  nsample = 0;
  ndig = 0;
}

dsp::HistUnpacker::~HistUnpacker ()
{
}

unsigned dsp::HistUnpacker::get_ndig () const
{
  if (ndig)
    return ndig;

  const_cast<HistUnpacker*>(this)->set_default_ndig();

  return ndig;
}

void dsp::HistUnpacker::set_default_ndig ()
{
  set_ndig( input->get_nchan() * input->get_npol() * input->get_ndim() );
}

/*! By default, there are two digitizers, one for each polarization */
void dsp::HistUnpacker::set_ndig (unsigned _ndig)
{
  if (ndig == _ndig)
    return;

  ndig = _ndig;
  resize ();
}

//! Set the number of time samples used to estimate undigitized power
void dsp::HistUnpacker::set_nsample (unsigned _nsample)
{
  if (nsample == _nsample)
    return;

  if (verbose)
    cerr << "dsp::HistUnpacker::set_nsample = " << _nsample << endl;

  nsample = _nsample;
  resize ();
}

/*! By default, there is no need to offset the output from each digitizer */
unsigned dsp::HistUnpacker::get_output_offset (unsigned idig) const
{
  if (input->get_state() == Signal::Analytic)
    return idig%2;

  return 0;
}

/*! By default, there is one digitizer for each polarization */
unsigned dsp::HistUnpacker::get_output_ipol (unsigned idig) const
{
  if (input->get_state() == Signal::Analytic)
    return idig/2;

  return idig;
}

/*! By default, there is only one frequency channel */
unsigned dsp::HistUnpacker::get_output_ichan (unsigned idig) const
{
  return 0;
}

void dsp::HistUnpacker::resize ()
{
  if (verbose)
    cerr << "dsp::HistUnpacker::resize ndig=" << ndig
         << " nsample=" << nsample << endl;

  histograms.resize (ndig);
  for (unsigned idig=0; idig<ndig; idig++)
    histograms[idig].resize (nsample);
}

void dsp::HistUnpacker::zero_histogram ()
{
  if (verbose)
    cerr << "dsp::HistUnpacker::zero_histogram" << endl;;

  for (unsigned ichan=0; ichan < histograms.size(); ichan++)
    for (unsigned ibin=0; ibin<histograms[ichan].size(); ibin++)
      histograms[ichan][ibin] = 0;
}

double dsp::HistUnpacker::get_histogram_mean (unsigned idig) const
{
  if (idig >= get_ndig())
    throw Error (InvalidParam, "dsp::HistUnpacker::get_histogram_mean",
		 "invalid idig=%d >= ndig=%d", idig, get_ndig());

  double ones = 0.0;
  double pts  = 0.0;

  for (unsigned ival=0; ival<nsample; ival++)
  {
    double samples = double (histograms[idig][ival]);
    ones += samples * double (ival);
    pts  += samples * double (nsample);
  }
  return ones/pts;
}

unsigned long dsp::HistUnpacker::get_histogram_total (unsigned idig) const
{
  if (idig >= get_ndig())
    throw Error (InvalidParam, "dsp::HistUnpacker::get_histogram_total",
                 "invalid idig=%d >= ndig=%d", idig, get_ndig());

  unsigned long nweights = 0;

  for (unsigned iwt=0; iwt<nsample; iwt++)
    nweights += histograms[idig][iwt];

  return nweights;
}

unsigned long* dsp::HistUnpacker::get_histogram (unsigned idig)
{
  if (idig >= get_ndig())
    throw Error (InvalidRange, "dsp::HistUnpacker::get_histogram",
                 "invalid idig=%d >= ndig=%d", idig, histograms.size());

  return &histograms[idig][0];
}



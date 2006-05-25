// #define _DEBUG 1

#include "dsp/HistUnpacker.h"

#include <iostream>

bool dsp::HistUnpacker::keep_histogram = true;

//! Null constructor
dsp::HistUnpacker::HistUnpacker (const char* _name) : Unpacker (_name)
{
  ndig = 2;
}

dsp::HistUnpacker::~HistUnpacker ()
{
}

/*! By default, there are two digitizers, one for each polarization */
void dsp::HistUnpacker::set_ndig (unsigned _ndig)
{
  if (ndig == _ndig)
    return;

  if (verbose)
    cerr << "dsp::HistUnpacker::set_ndig = " << _ndig << endl;

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
  return 0;
}

/*! By default, there is one digitizer for each polarization */
unsigned dsp::HistUnpacker::get_output_ipol (unsigned idig) const
{
  return idig;
}

/*! By default, there is only one frequency channel */
unsigned dsp::HistUnpacker::get_output_ichan (unsigned idig) const
{
  return 0;
}

void dsp::HistUnpacker::resize ()
{
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
  if ( idig >= get_ndig())
    throw Error (InvalidParam, "dsp::HistUnpacker::get_histogram_mean",
		 "invalid channel=%d", idig);

  double ones = 0.0;
  double pts  = 0.0;

  for (unsigned ival=0; ival<nsample; ival++) {
    double samples = double (histograms[idig][ival]);
    ones += samples * double (ival);
    pts  += samples * double (nsample);
  }
  return ones/pts;
}

unsigned long dsp::HistUnpacker::get_histogram_total (unsigned idig) const
{
  unsigned long nweights = 0;

  for (unsigned iwt=0; iwt<nsample; iwt++)
    nweights += histograms[idig][iwt];

  return nweights;
}


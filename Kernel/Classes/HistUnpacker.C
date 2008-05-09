/***************************************************************************
 *
 *   Copyright (C) 2005 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/HistUnpacker.h"
#include "dsp/WeightedTimeSeries.h"

#include <iostream>

using namespace std;

bool dsp::HistUnpacker::keep_histogram = true;

//! Null constructor
dsp::HistUnpacker::HistUnpacker (const char* _name) : Unpacker (_name)
{
  ndat_per_weight = 0;
  nstate = 0;
  nstate_internal = 0;

  ndig = 0;
  resize_needed = true;
}

dsp::HistUnpacker::~HistUnpacker ()
{
  // cerr << "dsp::HistUnpacker::~HistUnpacker" << endl;
}

void dsp::HistUnpacker::set_output (TimeSeries* _output)
{
  if (verbose)
    cerr << "dsp::HistUnpacker::set_output (" << _output << ")" << endl;

  Unpacker::set_output (_output);
  weighted_output = dynamic_cast<WeightedTimeSeries*> (_output);
}

//! Initialize and resize the output before calling unpack
void dsp::HistUnpacker::resize_output ()
{
  if (weighted_output)
  {
    weighted_output -> set_ndat_per_weight (get_ndat_per_weight());
    weighted_output -> set_nchan_weight (1);
    weighted_output -> set_npol_weight (input->get_npol());
  }

  output->resize ( input->get_ndat() );

  if (weighted_output)
    weighted_output -> neutral_weights ();
}

double dsp::HistUnpacker::get_optimal_variance ()
{
  throw Error (InvalidState, "dsp::HistUnpacker::get_optimal_variance",
               "not implemented");
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
  set_ndig( (input->get_nchan() * input->get_npol() * input->get_ndim())
            / get_ndim_per_digitizer() );
  if (verbose)
    cerr << "dsp::HistUnpacker::set_default_ndig ndig=" << ndig << endl;
}

/*! By default, there are two digitizers, one for each polarization */
void dsp::HistUnpacker::set_ndig (unsigned _ndig)
{
  if (ndig == _ndig)
    return;

  ndig = _ndig;
  resize_needed = true;
}

/*!
  By default, each digitizer samples a real-valued signal.

  In the case of dual-sideband downconversion, the in-phase and
  quadrature components are independently sampled, and ndim_per_digitizer=1.

  In the case of decimated output from a polyphase filterbank,
  the real and imaginary components of the complex values may be
  scaled and resampled together, and ndim_per_digitizer=2
*/
unsigned dsp::HistUnpacker::get_ndim_per_digitizer () const
{
  return 1;
}

//! Set the number of time samples per weight
void dsp::HistUnpacker::set_ndat_per_weight (unsigned _ndat_per_weight)
{
  if (verbose)
    cerr << "dsp::HistUnpacker::set_ndat_per_weight="
	 << _ndat_per_weight << endl;

  ndat_per_weight = _ndat_per_weight;
}

//! Set the number of possible states
void dsp::HistUnpacker::set_nstate (unsigned _nstate)
{
  if (nstate == _nstate)
    return;

  if (verbose)
    cerr << "dsp::HistUnpacker::set_nstate = " << _nstate << endl;

  nstate = _nstate;
  resize_needed = true;
}

//! Set the number of states in the internal representation of the histogram
void dsp::HistUnpacker::set_nstate_internal (unsigned _nstate)
{
  if (nstate_internal == _nstate)
    return;

  if (verbose)
    cerr << "dsp::HistUnpacker::set_nstate_internal = " << _nstate << endl;

  nstate_internal = _nstate;
  resize_needed = true;
}

/*! By default, there is no need to offset the output from each digitizer */
unsigned dsp::HistUnpacker::get_output_offset (unsigned idig) const
{
  if (input->get_state() == Signal::Analytic && get_ndim_per_digitizer() == 1)
    return idig%2;

  return 0;
}

/*! By default, there is one digitizer for each polarization */
unsigned dsp::HistUnpacker::get_output_ipol (unsigned idig) const
{
  if (input->get_state() == Signal::Analytic && get_ndim_per_digitizer() == 1)
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
         << " nstate=" << nstate << endl;

  histograms.resize( get_ndig() );
  for (unsigned idig=0; idig<ndig; idig++)
    histograms[idig].resize( get_nstate_internal() );

  resize_needed = false;
}

unsigned dsp::HistUnpacker::get_nstate_internal () const
{
  if (nstate_internal)
    return nstate_internal;
  else
    return nstate;
}

void dsp::HistUnpacker::zero_histogram ()
{
  if (verbose)
    cerr << "dsp::HistUnpacker::zero_histogram" << endl;;

  for (unsigned ichan=0; ichan < histograms.size(); ichan++)
    for (unsigned ibin=0; ibin<histograms[ichan].size(); ibin++)
      histograms[ichan][ibin] = 0;
}

void dsp::HistUnpacker::get_histogram (std::vector<unsigned long>& hist,
				       unsigned idig) const
{
  if (idig >= get_ndig())
    throw Error (InvalidParam, "dsp::HistUnpacker::get_histogram",
		 "invalid idig=%d >= ndig=%d", idig, get_ndig());

  hist = histograms[idig];
}

double dsp::HistUnpacker::get_histogram_mean (unsigned idig) const
{
  if (idig >= get_ndig())
    throw Error (InvalidParam, "dsp::HistUnpacker::get_histogram_mean",
		 "invalid idig=%d >= ndig=%d", idig, get_ndig());

  double ones = 0.0;
  double pts  = 0.0;

  for (unsigned ival=0; ival<nstate; ival++)
  {
    double samples = double (histograms[idig][ival]);
    ones += samples * double (ival);
    pts  += samples * double (nstate);
  }
  return ones/pts;
}

unsigned long dsp::HistUnpacker::get_histogram_total (unsigned idig) const
{
  if (idig >= get_ndig())
    throw Error (InvalidParam, "dsp::HistUnpacker::get_histogram_total",
                 "invalid idig=%d >= ndig=%d", idig, get_ndig());

  unsigned long nweights = 0;

  for (unsigned iwt=0; iwt<nstate; iwt++)
    nweights += histograms[idig][iwt];

  return nweights;
}

unsigned long* dsp::HistUnpacker::get_histogram (unsigned idig)
{
  if (resize_needed)
    resize ();

  if (idig >= get_ndig())
    throw Error (InvalidRange, "dsp::HistUnpacker::get_histogram",
                 "invalid idig=%d >= ndig=%d", idig, histograms.size());

  return &histograms[idig][0];
}

const unsigned long* dsp::HistUnpacker::get_histogram (unsigned idig) const
{
  return const_cast<HistUnpacker*>(this)->get_histogram(idig);
}



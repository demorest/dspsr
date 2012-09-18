/***************************************************************************
 *
 *   Copyright (C) 2005-2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/HistUnpacker.h"

#include <iostream>
#include <assert.h>

using namespace std;

bool dsp::HistUnpacker::keep_histogram = true;

//! Null constructor
dsp::HistUnpacker::HistUnpacker (const char* _name) : Unpacker (_name)
{
  nstate = 0;
  nstate_internal = 0;

  ndig = 0;
  resize_needed = true;
}

dsp::HistUnpacker::~HistUnpacker ()
{
  // cerr << "dsp::HistUnpacker::~HistUnpacker" << endl;
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
  if (!input)
    throw Error (InvalidState, "dsp::HistUnpacker::set_default_ndig",
                 "input not set");

  set_ndig( (input->get_nchan() * input->get_npol() * input->get_ndim())
            / get_ndim_per_digitizer() );
  if (verbose)
    cerr << "dsp::HistUnpacker::set_default_ndig ndig=" << ndig << endl;

  resize();
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
  ndim per digitizer is 1 for both real data and complex data that have
  been created by quadrature down conversion, where real and imaginary
  components are independently sampled.

  ndim per digitizer is 2 for complex data that have been created by
  digitial electronics, such as a polyphase filterbank, especially
  if the real and imaginary components are scaled and resampled together.
*/
unsigned dsp::HistUnpacker::get_ndim_per_digitizer () const
{
  return 1;
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
    cerr << "dsp::HistUnpacker::resize this=" << this << " ndig=" << get_ndig()
         << " nstate=" << nstate
         << " internal=" << get_nstate_internal() << endl;

  histograms.resize( get_ndig() );
  for (unsigned idig=0; idig<get_ndig(); idig++)
    histograms[idig].resize( get_nstate_internal() );

  resize_needed = false;

  zero_histogram();
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

  // if the input is not set, there may be no way to resize
  if (!input)
    return;

  if (resize_needed)
    resize ();

  for (unsigned idig=0; idig < histograms.size(); idig++)
  {
    if (verbose)
      cerr << "dsp::HistUnpacker::zero_histogram idig=" << idig
           << " size=" << histograms[idig].size() << endl;
    for (unsigned ibin=0; ibin<histograms[idig].size(); ibin++)
      histograms[idig][ibin] = 0;
  }
}

void dsp::HistUnpacker::get_histogram (std::vector<unsigned long>& hist,
				       unsigned idig) const
{
  if (idig >= histograms.size())
    throw Error (InvalidParam, "dsp::HistUnpacker::get_histogram",
		 "invalid idig=%d >= ndig=%d", idig, histograms.size());

  if (verbose)
    cerr << "dsp::HistUnpacker::get_histogram this=" << this
         << " idig=" << idig << " size=" << histograms[idig].size() << endl;

  hist = histograms[idig];
}

double dsp::HistUnpacker::get_histogram_mean (unsigned idig) const
{
  if (idig >= histograms.size())
    throw Error (InvalidParam, "dsp::HistUnpacker::get_histogram_mean",
		 "invalid idig=%d >= ndig=%d", idig, histograms.size());

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

unsigned long* dsp::HistUnpacker::get_histogram (unsigned idig, unsigned expect)
{
  if (verbose)
    cerr << "dsp::HistUnpacker::get_histogram idig=" << idig
         << " expect=" << expect << endl;

  if (resize_needed)
    resize ();

  if (idig >= histograms.size())
    throw Error (InvalidRange, "dsp::HistUnpacker::get_histogram",
                 "invalid idig=%d >= ndig=%d", idig, histograms.size());

  if (expect && histograms[idig].size() != expect)
    throw Error (InvalidState, "dsp::HistUnpacker::get_histogram",
                 "this=%x histograms[%d].size = %u != expected = %u", this,
                 idig, histograms[idig].size(), expect);


  return &histograms[idig][0];
}

const unsigned long* dsp::HistUnpacker::get_histogram (unsigned idig) const
{
  return const_cast<HistUnpacker*>(this)->get_histogram(idig);
}


//! Combine results with another operation
void dsp::HistUnpacker::combine (const Operation* other)
{
  Operation::combine (other);

  if (histograms.size() == 0)
    return;

  const HistUnpacker* like = dynamic_cast<const HistUnpacker*>( other );
  if (!like)
    return;

  if (like->histograms.size() == 0)
    return;

  if (verbose)
    cerr << "dsp::HistUnpacker::combine this=" << this
	 << " like=" << like << endl;

  assert (histograms.size() == like->histograms.size());

  for (unsigned idig=0; idig < histograms.size(); idig++)
  {
    assert (histograms[idig].size() == like->histograms[idig].size());
    for (unsigned ibin=0; ibin<histograms[idig].size(); ibin++)
      histograms[idig][ibin] += like->histograms[idig][ibin];
  }
}

//! Combine results with another operation
void dsp::HistUnpacker::reset ()
{
  Operation::reset ();
  zero_histogram ();
}

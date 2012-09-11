/***************************************************************************
 *
 *   Copyright (C) 2002-2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/PhaseSeries.h"
#include "dsp/dspExtension.h"
#include "dsp/Memory.h"

#include "Pulsar/Predictor.h"
#include "Pulsar/Parameters.h"

using namespace std;

void dsp::PhaseSeries::init ()
{
  if (verbose)
    cerr << "dsp::PhaseSeries::init this=" << this << endl;

  integration_length = 0;
  ndat_total = 0;
  ndat_expected = 0;

  folding_period = 0;
  reference_phase = 0;
  require_equal_sources = false;
  require_equal_rates = false;

  hits_memory = Memory::get_manager();

  hits = 0;
  hits_nchan = 1;
  hits_size = 0;
}

dsp::PhaseSeries::PhaseSeries () : TimeSeries()
{
  init ();
}

dsp::PhaseSeries::PhaseSeries (const PhaseSeries& profile) : TimeSeries ()
{
  init ();
  operator= (profile);
}

dsp::PhaseSeries::~PhaseSeries ()
{
  if (verbose)
    cerr << "dsp::PhaseSeries::~PhaseSeries this=" << this << " hits=" << hits << endl;
  if (hits)
    hits_memory->do_free(hits);
  hits = 0;
}

void dsp::PhaseSeries::set_hits_memory (Memory* m)
{
  if (hits_memory && hits)
    resize_hits (0);

  hits_memory = m;
}

const dsp::Memory* dsp::PhaseSeries::get_hits_memory () const
{ 
  return hits_memory;
}


dsp::PhaseSeries* dsp::PhaseSeries::clone() const
{
  if (verbose)
    cerr << "dsp::PhaseSeries::clone" << endl;

  return new PhaseSeries (*this);
}

//! Set the number of phase bins into which data will be PhaseSeriesed
void dsp::PhaseSeries::resize (int64_t nbin)
{
  if (verbose)
    cerr << "dsp::PhaseSeries::resize nbin=" << nbin << endl;
  TimeSeries::resize (nbin);
  resize_hits (nbin);
}

//! Resize the hits arary
void dsp::PhaseSeries::resize_hits (int64_t nbin)
{
  if (verbose)
    cerr << "dsp::PhaseSeries::resize_hits nchan=" << get_nchan()
         << " nbin=" << nbin << " hits_nchan=" << hits_nchan << endl;

  if (hits)
  {
    if (verbose)
      cerr << "dsp::PhaseSeries::resize_hits Memory::do_free()" << endl;
    hits_memory->do_free (hits);
  }

  hits_size = nbin * hits_nchan * sizeof (unsigned);

  if (verbose)
    cerr << "dsp::PhaseSeries::resize_hits Memory::do_allocate (" 
         << hits_size << ")" << endl;
  hits = (unsigned *) hits_memory->do_allocate (hits_size);

}

//! Set the period at which to fold data (in seconds)
void dsp::PhaseSeries::set_folding_period (double _folding_period)
{
  folding_period = _folding_period;
  folding_predictor = 0;
  pulsar_ephemeris = 0;
}

//! Get the average folding period
double dsp::PhaseSeries::get_folding_period () const
{
  if (folding_predictor)  {
    MJD mid_time = get_mid_time();
    if (verbose)
      cerr << "dsp::PhaseSeries::get_folding_period mid_time=" 
           << mid_time.printdays(30) << endl;
    double freq = folding_predictor->frequency( get_mid_time() );
    if (verbose) {
      cerr.precision(30);
      cerr << "dsp::PhaseSeries::get_folding_period frequency=" << freq << endl;
    }
    return 1.0 / freq;
  }
  else
    return folding_period;
}

void dsp::PhaseSeries::set_pulsar_ephemeris (const Pulsar::Parameters* eph)
{
  pulsar_ephemeris = eph;
}

void dsp::PhaseSeries::set_folding_predictor (const Pulsar::Predictor* p)
{
  if (verbose)
    cerr << "dsp::PhaseSeries::set_folding_predictor " << p << endl;
  folding_predictor = p;
  folding_period = 0.0;
}

//! Inquire the phase polynomial(s) with which to fold data
const Pulsar::Predictor* dsp::PhaseSeries::get_folding_predictor () const try
{
  return folding_predictor;
}
catch (Error& error)
{
  throw error += "dsp::PhaseSeries::get_folding_predictor";
}

bool dsp::PhaseSeries::has_folding_predictor () const
{
  return folding_predictor;
}

//! Inquire the ephemeris used to fold the data
const Pulsar::Parameters* dsp::PhaseSeries::get_pulsar_ephemeris () const try
{
  return pulsar_ephemeris;
}
catch (Error& error)
{
  throw error += "dsp::PhaseSeries::get_pulsar_ephemeris";
}

bool dsp::PhaseSeries::has_pulsar_ephemeris () const
{
  return pulsar_ephemeris;
}

unsigned dsp::PhaseSeries::get_hit (unsigned ibin, unsigned ichan) const
{
  if (ichan >= hits_nchan)
    throw Error (InvalidParam, "PhaseSeries::get_hit", "ichan >= hits_nchan"); 

  return hits[ichan * get_nbin() + ibin];
}

unsigned* dsp::PhaseSeries::get_hits (unsigned ichan)
{

  if (ichan >= hits_nchan)
    throw Error (InvalidParam, "PhaseSeries::get_hits", "ichan >= hits_nchan");

  return &(hits[ichan * get_nbin()]);
}

const unsigned* dsp::PhaseSeries::get_hits (unsigned ichan) const
{
  if (ichan >= hits_nchan)
    throw Error (InvalidParam, "PhaseSeries::get_hits", "ichan >= hits_nchan");

  return &(hits[ichan * get_nbin()]);
}

//! Get the mid-time of the integration
MJD dsp::PhaseSeries::get_mid_time (bool phased) const
{
  if (verbose)
  {
    cerr << "PhaseSeries::get_mid_time start=" 
      << start_time << " end=" << end_time << endl;
  }

  MJD midtime = 0.5 * (start_time + end_time);

  if (!phased)
    return midtime;

  if (folding_predictor)
  {
    // truncate midtime to the nearest pulse phase = reference_phase
    Phase phase = folding_predictor->phase(midtime).Floor() + reference_phase;
    midtime = folding_predictor->iphase (phase, &midtime);
  }

  if (folding_period)
  {
    double phase = reference_phase + 
      fmod (midtime.in_seconds(), folding_period)/folding_period;
    midtime -= phase * folding_period;
  }

  return midtime;
}

//! Reset all phase bin totals to zero
void dsp::PhaseSeries::zero () try
{
  if (verbose)
    cerr << "PhaseSeries::zero" << endl;

  integration_length = 0.0;
  ndat_total = 0;

  set_hits (0);
  TimeSeries::zero ();

  if (verbose)
    cerr << "PhaseSeries::zero exit" << endl;
}
 catch (Error& error)
   {
     throw error += "dsp::PhaseSeries::zero";
   }

void dsp::PhaseSeries::copy_configuration (const Observation* copy)
{
  if (verbose)
    cerr << "dsp::PhaseSeries::copy_configuration" << endl;

  TimeSeries::copy_configuration (copy);

  const PhaseSeries* like = dynamic_cast<const PhaseSeries*>(copy);
  if (like)
  {
    if (verbose)
      cerr << "dsp::PhaseSeries::copy_configuration copy PhaseSeries" << endl;
    copy_attributes (like);  
  }
}

void dsp::PhaseSeries::copy_attributes (const PhaseSeries* copy)
{
  reference_phase    = copy->reference_phase;
  integration_length = copy->integration_length;
  ndat_total         = copy->ndat_total;
  ndat_expected      = copy->ndat_expected;

  end_time           = copy->end_time;
  folding_period     = copy->folding_period;

  if (verbose)
    cerr << "dsp::PhaseSeries::copy_attributes hits_size=" << hits_size 
         << " copy->hits_size=" << copy->hits_size 
         << " copy->hits_nchan=" << copy->hits_nchan << endl;

  if (hits_size < copy->hits_size)
  {
    if (hits)
      hits_memory->do_free (hits);

    hits = (unsigned *) hits_memory->do_allocate (copy->hits_size);
    if (verbose)
      cerr << "dsp::PhaseSeries::copy_attributes hits_memory::do_allocate(" 
           << copy->hits_size << ")" << endl;
  }

  // if both hits arrays are on host memory
  if (get_hits_memory()->on_host() && copy->get_hits_memory()->on_host())
    hits_memory->do_copy ( hits, copy->hits, copy->hits_size);

  zeroed_data = copy->zeroed_data;
  hits_nchan = copy->hits_nchan;
  hits_size = copy->hits_size;

  if (copy->folding_predictor)
    folding_predictor = copy->folding_predictor->clone();
  else
    folding_predictor = 0;

  if (copy->pulsar_ephemeris)
    pulsar_ephemeris = copy->pulsar_ephemeris->clone();
  else
    pulsar_ephemeris = 0;
}

//! Set the hits in all bins
void dsp::PhaseSeries::set_hits (unsigned value)
{
  if (verbose)
    cerr << "PhaseSeries::set_hits(" << value << ")" << endl;

  if (value == 0)
    hits_memory->do_zero(hits, hits_size);
  else
  {

    const unsigned nhits = get_nbin() * hits_nchan;
    for (unsigned ihit=0; ihit<nhits; ihit++)
      hits[ihit] = value;
  }
}

bool dsp::PhaseSeries::mixable (const Observation& obs, unsigned nbin,
				int64_t istart, int64_t fold_ndat)
{
  MJD obsStart = obs.get_start_time() + double (istart) / obs.get_rate();

  if (verbose)
    cerr << "PhaseSeries::mixable start mix=" << obsStart.printdays(8)
	 << " cur=" << get_start_time().printdays(8) << endl;

  MJD obsEnd;

  // if fold_ndat is not specified, fold to the end of the Observation
  // (works also for special case of adding dsp::PhaseSeriess together;
  // where using ndat=nbin would not make sense)
  if (fold_ndat == 0)
    obsEnd = obs.get_end_time();
  else
    obsEnd = obsStart + double (fold_ndat) / obs.get_rate();

  if (integration_length == 0.0)
  {
    // the integration is currently empty; prepare for integration

    if (verbose)
      cerr << "PhaseSeries::mixable reset" << endl;

    Observation::operator = (obs);
    if (verbose)
      cerr << "dsp::PhaseSeries::mixable rate=" << get_rate() << endl;

    const TimeSeries* series = dynamic_cast<const TimeSeries*> (&obs);
    if (series)
    {
      if (verbose)
        cerr << "dsp::PhaseSeries::mixable calling set_order" << endl;
      set_order( series->get_order() );
      if (verbose)
        cerr << "dsp::PhaseSeries::mixable calling set_zeroed_data" << endl;
      set_zeroed_data( series->get_zeroed_data() );
      if (get_zeroed_data())
        set_hits_nchan( series->get_nchan() );
    }

    end_time = obsEnd;
    start_time = obsStart;

    /*
      the integration length may be zero only because all of the samples
      have been dropped - maintain the record of dropped samples
    */
    uint64_t backup_ndat_total = ndat_total;

    if (verbose)
      cerr << "dsp::PhaseSeries::mixable calling resize(" << nbin << ")" << endl;
    resize (nbin);
    if (verbose)
      cerr << "dsp::PhaseSeries::mixable calling zero()" << endl;
    zero ();

    ndat_total = backup_ndat_total;

    return true;
  }

  if (!combinable (obs)) {
    cerr << "PhaseSeries::mixable differing observations" << endl;
    return false;
  }

  if (get_nbin() != nbin) {
    cerr << "PhaseSeries::mixable nbin=" << get_nbin() <<" != "<< nbin <<endl;
    return false;
  }

  end_time = std::max (end_time, obsEnd);
  start_time = std::min (start_time, obsStart);

  if (verbose)
    cerr << "PhaseSeries::mixable combine start=" << start_time.printdays(8)
	 << " end=" << end_time.printdays(8) << endl;

  return true;
}

dsp::PhaseSeries& 
dsp::PhaseSeries::operator = (const PhaseSeries& prof) try
{
  if (this == &prof)
    return *this;

  if (verbose)
    cerr << "dsp::PhaseSeries::operator = call TimeSeries::operator =" << endl;
  TimeSeries::operator = (prof);

  if (verbose)
    cerr << "dsp::PhaseSeries::operator = copy attributes" << endl;

  copy_attributes (&prof);

  return *this;
}
 catch (Error& error)
   {
     throw error += "dsp::PhaseSeries::operator =";
   }

void dsp::PhaseSeries::combine (const PhaseSeries* prof) try
{
  if (verbose)
    cerr << "dsp::PhaseSeries::combine"
            " this=" << this << " that=" << prof << endl;

  if (!prof || prof->get_nbin() == 0)
    return;

  if (verbose)
    cerr << "dsp::PhaseSeries::combine length add=" 
         << prof->integration_length 
         << " current=" << integration_length << endl;

  if (!integration_length)
  {
    if (verbose)
      cerr << "dsp::PhaseSeries::combine this is empty" << endl;

    *this = *prof;
    return;
  }

  if (!mixable (*prof, prof->get_nbin()))
    throw Error (InvalidParam, "PhaseSeries::combine",
		 "PhaseSeries !mixable");

  TimeSeries::operator += (*prof);

  const unsigned nhits = get_nbin() * hits_nchan;
  for (unsigned ihit=0; ihit<nhits; ihit++)
    hits[ihit] += prof->hits[ihit];
  
  integration_length += prof->integration_length;
  ndat_total += prof->ndat_total;

  if (!ndat_expected)
    ndat_expected = prof->ndat_expected;
}
 catch (Error& error)
   {
     throw error += "dsp::PhaseSeries::combine";
   }

//! Return the total number of time samples
uint64_t dsp::PhaseSeries::get_ndat_total () const
{
  return ndat_total;
}

uint64_t dsp::PhaseSeries::get_ndat_folded () const
{
  uint64_t folded = 0;
  const unsigned nhits = get_nbin() * hits_nchan;
  for (unsigned ihit=0; ihit<nhits; ihit++)
    folded += hits[ihit];

  return folded;
}


//! Return the number of time samples folded into the profiles
uint64_t dsp::PhaseSeries::get_ndat_folded (unsigned ichan) const
{
  uint64_t folded = 0;
  const unsigned nbin = get_nbin();

  if (ichan > hits_nchan)
    throw Error (InvalidParam, "PhaseSeries::get_ndat_folded", "ichan >= hits_nchan");

  for (unsigned ibin=0; ibin<nbin; ibin++)
    folded += hits[ichan * nbin + ibin];
  
  return folded;
}

//! Set the expected number of time samples
void dsp::PhaseSeries::set_ndat_expected (uint64_t n)
{
  ndat_expected = n;
}

//! Return the expected number of time samples
uint64_t dsp::PhaseSeries::get_ndat_expected () const
{
  return ndat_expected;
}

void dsp::PhaseSeries::set_extensions (Extensions* ext)
{
  extensions = ext;
}

const dsp::Extensions* dsp::PhaseSeries::get_extensions () const
{
  return extensions;
}

dsp::Extensions* dsp::PhaseSeries::get_extensions ()
{
  return extensions;
}

bool dsp::PhaseSeries::has_extensions () const
{
  return extensions;
}


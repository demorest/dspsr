/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/PhaseSeries.h"

#include "Pulsar/Predictor.h"
#include "psrephem.h"

using namespace std;

dsp::PhaseSeries::PhaseSeries () : TimeSeries()
{
  integration_length = 0;
  folding_period = 0;
  reference_phase = 0;
}

dsp::PhaseSeries::PhaseSeries (const PhaseSeries& profile) : TimeSeries ()
{
  operator= (profile);
}

dsp::PhaseSeries::~PhaseSeries () { }

//! Set the number of phase bins into which data will be PhaseSeriesed
void dsp::PhaseSeries::resize (int64 nbin)
{
  TimeSeries::resize (nbin);
  hits.resize (nbin);
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
  if (folding_predictor)
    return 1.0 / folding_predictor->frequency( get_mid_time() );
  else
    return folding_period;
}

//! Set the pulsar ephemeris used to fold with.  User must also supply the Pulsar::Predictor that was generated from the ephemeris and used for folding
void dsp::PhaseSeries::set_pulsar_ephemeris(const psrephem* _pulsar_ephemeris, const Pulsar::Predictor* _folding_predictor)
{
  pulsar_ephemeris = _pulsar_ephemeris;
  folding_predictor = _folding_predictor;
  folding_period = 0.0;
}

//! Inquire the phase polynomial(s) with which to fold data
const Pulsar::Predictor* dsp::PhaseSeries::get_folding_predictor () const
{
  return folding_predictor.ptr();
}

//! Inquire the ephemeris used to fold the data
const psrephem* dsp::PhaseSeries::get_pulsar_ephemeris () const
{
  return pulsar_ephemeris.ptr();
}

//! Get the mid-time of the integration
MJD dsp::PhaseSeries::get_mid_time () const
{
  MJD midtime = 0.5 * (start_time + end_time);

  if (folding_predictor) {
    // truncate midtime to the nearest pulse phase = reference_phase
    Phase phase = folding_predictor->phase(midtime).Floor() + reference_phase;
    midtime = folding_predictor->iphase(phase);
  }

  if (folding_period) {
    double phase = reference_phase + 
      fmod (midtime.in_seconds(), folding_period)/folding_period;
    midtime -= phase * folding_period;
  }

  return midtime;
}

//! Reset all phase bin totals to zero
void dsp::PhaseSeries::zero ()
{
  if (verbose)
    cerr << "PhaseSeries::zero" << endl;

  integration_length = 0.0;
  set_hits (0);
  TimeSeries::zero ();
}

//! Set the hits in all bins
void dsp::PhaseSeries::set_hits (unsigned value)
{
  for (unsigned ipt=0; ipt<hits.size(); ipt++)
    hits[ipt] = value;
}

bool dsp::PhaseSeries::mixable (const Observation& obs, unsigned nbin,
				int64 istart, int64 fold_ndat)
{
  static MJD st = obs.get_start_time(); 
  MJD obsStart = obs.get_start_time() + double (istart) / obs.get_rate();

  if (verbose)
    cerr << "PhaseSeries::mixable"
         << "\n  mix->start=" << (obsStart-st).in_seconds()
	 << "\n istart=" << istart
	 << "\n this->start=" << (get_start_time()-st).in_seconds() << endl;

  MJD obsEnd;

  // if fold_ndat is not specified, fold to the end of the Observation
  // (works also for special case of adding dsp::PhaseSeriess together;
  // where using ndat=nbin would not make sense)
  if (fold_ndat == 0)
    obsEnd = obs.get_end_time();
  else
    obsEnd = obsStart + double (fold_ndat) / obs.get_rate();

  if (integration_length == 0.0) {

    // the integration is currently empty; prepare for integration

    if (verbose)
      cerr << "PhaseSeries::mixable reset" << endl;

    Observation::operator = (obs);
    if( verbose )
      fprintf(stderr,"dsp::PhaseSeries::mixable() has acquired rate=%f\n",get_rate());

    end_time = obsEnd;
    start_time = obsStart;

    resize (nbin);
    zero ();

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
    cerr << "PhaseSeries::mixable combine start=" << start_time
	 << " end=" << end_time << endl;

  return true;
}

dsp::PhaseSeries& 
dsp::PhaseSeries::operator = (const PhaseSeries& prof)
{
  if (this == &prof)
    return *this;

  if (verbose)
    cerr << "dsp::PhaseSeries::operator = call TimeSeries::operator =" << endl;
  TimeSeries::operator = (prof);

  if (verbose)
    cerr << "dsp::PhaseSeries::operator = copy attributes" << endl;

  integration_length = prof.integration_length;
  end_time           = prof.end_time;
  folding_period     = prof.folding_period;
  folding_predictor     = prof.folding_predictor;
  pulsar_ephemeris   = prof.pulsar_ephemeris;
  hits               = prof.hits;

  return *this;
}

dsp::PhaseSeries&
dsp::PhaseSeries::operator += (const PhaseSeries& prof)
{
  if (!mixable (prof, prof.get_nbin()))
    throw Error (InvalidParam, "PhaseSeries::operator+=",
		 "PhaseSeries !mixable");

  TimeSeries::operator += (prof);

  for (unsigned ibin=0; ibin<hits.size(); ibin++)
    hits[ibin] += prof.hits[ibin];

  integration_length += prof.integration_length;

  return *this;
}



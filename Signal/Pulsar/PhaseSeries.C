#include "dsp/PhaseSeries.h"
#include "polyco.h"
#include "genutil.h"

dsp::PhaseSeries::PhaseSeries ()
{
  integration_length = 0;
  folding_period = 0;
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
  folding_polyco = 0;
}

//! Get the average folding period
double dsp::PhaseSeries::get_folding_period () const
{
  if (folding_polyco)
    return folding_polyco->get_refperiod();
  else
    return folding_period;
}

//! Set the phase polynomial(s) with which to fold data
void dsp::PhaseSeries::set_folding_polyco (const polyco* _folding_polyco)
{
  folding_polyco = _folding_polyco;
  folding_period = 0.0;
}

//! Set the phase polynomial(s) with which to fold data
const polyco* dsp::PhaseSeries::get_folding_polyco () const
{
  if (!folding_polyco)
    throw_str ("PhaseSeries::get_folding_polyco polyco not set");

  return folding_polyco;
}

//! Get the mid-time of the integration
MJD dsp::PhaseSeries::get_mid_time () const
{
  MJD midtime = 0.5 * (start_time + end_time);

  if (folding_polyco) {
    // truncate midtime to the nearest pulse phase zero
    Phase phase = folding_polyco->phase(midtime).Floor();
    midtime = folding_polyco->iphase(phase);
  }

  if (folding_period) {
    double phase = fmod (midtime.in_seconds(), folding_period)/folding_period;
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

  unsigned ipt=0; 
  for (ipt=0; ipt<hits.size(); ipt++)
    hits[ipt]=0;

  TimeSeries::zero ();
}


bool dsp::PhaseSeries::mixable (const Observation& obs, int nbin,
				     int64 istart, int64 fold_ndat)
{
  MJD obsStart = obs.get_start_time() + double (istart) / obs.get_rate();

  if (verbose)
    cerr << "PhaseSeries::mixable"
         << "\n  mix->start=" << obsStart
	 << "\n this->start=" << get_start_time() << endl;

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
    cerr << "PhaseSeries::mixable nbin="<<get_nbin()<<" != "<<nbin<<endl;
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

  TimeSeries::operator = (prof);

  integration_length = prof.integration_length;
  end_time           = prof.end_time;
  folding_period     = prof.folding_period;
  folding_polyco     = prof.folding_polyco;
  hits               = prof.hits;

  return *this;
}

dsp::PhaseSeries&
dsp::PhaseSeries::operator += (const PhaseSeries& prof)
{
  if (!mixable (prof, prof.get_nbin()))
    throw_str ("PhaseSeries::operator+= !mixable");

  TimeSeries::operator += (prof);

  for (unsigned ibin=0; ibin<hits.size(); ibin++)
    hits[ibin] += prof.hits[ibin];

  integration_length += prof.integration_length;

  return *this;
}


#include "Fold.h"
#include "polyco.h"

void dsp::Fold::init ()
{
  integration_length = 0;
  folding_period = 0;
}

//! Set the number of phase bins into which data will be folded
void dsp::Fold::set_nbin (int nbin)
{
  hits.resize(nbin);
}

//! Set the number of phase bins into which data will be folded
int dsp::Fold::get_nbin () const
{
  return (int) hits.size();
}

//! Set the period at which to fold data (in seconds)
void dsp::Fold::set_folding_period (double _folding_period)
{
  folding_period = _folding_period;
  folding_polyco = 0;
}

//! Get the average folding period
double dsp::Fold::get_folding_period () const
{
  if (folding_polyco)
    return folding_polyco->get_refperiod();
  else
    return folding_period;
}

//! Set the phase polynomial(s) with which to fold data
void dsp::Fold::set_folding_polyco (polyco* _folding_polyco)
{
  folding_polyco = _folding_polyco;
  folding_period = 0.0;
}

//! Get the mid-time of the integration
MJD dsp::Fold::get_midtime () const
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

//! Get the number of seconds integrated into the profile(s)
double dsp::Fold::get_integration_length () const
{
  return integration_length;
}

//! Reset all phase bin totals to zero
void dsp::Fold::zero ()
{
  integration_length = 0.0;

  unsigned ipt=0; 
  for (ipt=0; ipt<hits.size(); ipt++)
    hits[ipt]=0;

  for (ipt=0; ipt<amps.size(); ipt++)
    amps[ipt]=0.0;
}

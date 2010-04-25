/***************************************************************************
 *
 *   Copyright (C) 2003-2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SubFold.h"
#include "dsp/SignalPath.h"

#include "dsp/PhaseSeriesUnloader.h"

#include "Error.h"

using namespace std;

dsp::SubFold::SubFold ()
{
  built = false;
}

dsp::SubFold::~SubFold ()
{
  if (verbose)
    cerr << "dsp::SubFold::~SubFold" << endl;
}

dsp::Fold* dsp::SubFold::clone () const
{
  return new SubFold(*this);
}

//! Set verbosity ostream
void dsp::SubFold::set_cerr (std::ostream& os) const
{
  Operation::set_cerr (os);
  divider.set_cerr (os);
}

//! Set the file unloader
void dsp::SubFold::set_unloader (dsp::PhaseSeriesUnloader* _unloader)
{
  if (verbose)
    cerr << "dsp::SubFold::set_unloader this=" << this 
         << " unloader=" << _unloader << endl;

  unloader = _unloader;
}

//! Get the file unloader
dsp::PhaseSeriesUnloader* dsp::SubFold::get_unloader () const
{
  return unloader;
}


//! Set the start time from which to begin counting sub-integrations
void dsp::SubFold::set_start_time (const MJD& start_time)
{
  cerr << "dsp::SubFold::set_start_time" << endl;
  divider.set_start_time (start_time);
}

//! Set the interval over which to fold each sub-integration (in seconds)
void dsp::SubFold::set_subint_seconds (double subint_seconds)
{
  divider.set_seconds (subint_seconds);
}

//! Set the number of pulses to fold into each sub-integration
void dsp::SubFold::set_subint_turns (unsigned subint_turns)
{
  divider.set_turns (subint_turns);
}

void dsp::SubFold::set_fractional_pulses (bool flag)
{
  divider.set_fractional_pulses (flag);
}

void dsp::SubFold::prepare ()
{
  if (verbose)
    cerr << "dsp::SubFold::prepare call Fold::prepare" << endl;

  Fold::prepare ();

  // if unspecified, the first TimeSeries to be folded will define the
  // start time from which to begin cutting up the observation
  if (divider.get_start_time() == MJD::zero)
  {
    if (verbose)
      cerr << "dsp::SubFold::prepare set divider start time=" 
	   << input->get_start_time() << endl;

    if (input->get_start_time() == MJD::zero)
      throw Error (InvalidState, "dsp::SubFold::prepare",
		   "input start time not set");

    divider.set_start_time (input->get_start_time());
  }

  if (has_folding_predictor() && divider.get_turns())
    divider.set_predictor (get_folding_predictor());

  built = true;
}


void dsp::SubFold::transformation () try
{
  if (verbose)
    cerr << "dsp::SubFold::transformation" << endl;

  if (divider.get_turns() == 0 && divider.get_seconds() == 0.0)
    throw Error (InvalidState, "dsp::SubFold::tranformation",
		 "sub-integration length not specified");

  if (!built)
    prepare ();

  // flag that the input TimeSeries contains data for another sub-integration
  bool more_data = true;
  bool first_division = true;

  while (more_data)
  {
    divider.set_bounds( get_input() );

    if (!divider.get_fractional_pulses())
      get_output()->set_ndat_expected( divider.get_division_ndat() );

    more_data = divider.get_in_next ();

    if (divider.get_new_division())
    {
      /* A new division has been started and there is still data in
	 the current integration.  This is a sign that the current
	 input comes from uncontiguous data, which can arise when
	 processing in parallel. */

      unload_partial ();
    }

    if (!divider.get_is_valid())
      continue;

    Fold::transformation ();

    if (!divider.get_end_reached())
      continue;

    if (first_division)
    {
      /* When the end of the first division is reached, it is not 100%
         certain that a complete sub-integration is available */
      unload_partial ();
      first_division = false;
      continue;
    }

    if (verbose)
      cerr << "dsp::SubFold::transformation sub-integration completed" << endl;

    PhaseSeries* result = get_result ();

    complete.send (result);

    if (unloader && keep(result))
    {
      if (verbose)
	cerr << "dsp::SubFold::transformation this=" << this
             << " unloader=" << unloader.get() << endl;

      unloader->unload (result);
    }

    zero_output ();
  }
}
catch (Error& error)
{
  throw error += "dsp::SubFold::transformation";
}


/*! sets the Fold::idat_start and Fold::ndat_fold attributes */
void dsp::SubFold::set_limits (const Observation* input)
{
  idat_start = divider.get_idat_start ();
  ndat_fold = divider.get_ndat ();
}

void dsp::SubFold::finish () try
{
  if (!get_result()->get_integration_length())
  {
    if (verbose)
      cerr << "dsp::SubFold::finish unload_partial" << endl;

    unload_partial ();
  }

  if (unloader)
  {
    if (verbose)
      cerr << "dsp::SubFold::finish call unloader finish" << endl;

    unloader->finish ();
  }
}
catch (Error& error)
{
  throw error += "dsp::SubFold::finish";
}

void dsp::SubFold::unload_partial () try
{
  if (verbose)
    cerr << "dsp::SubFold::unload_partial to callback" << endl;

  PhaseSeries* result = get_result ();

  partial.send (result);

  if (unloader)
  {
    if (verbose)
      cerr << "dsp::SubFold::unload_partial this=" << this
           << " unloader=" << unloader.get() << endl;

    unloader->partial (result);
  }

  zero_output ();
}
catch (Error& error)
{
  throw error += "dsp::SubFold::finish";
}

#define SIGNAL_PATH

void dsp::SubFold::zero_output ()
{
#ifdef SIGNAL_PATH
  SignalPath* path = 0;

  if (output->has_extensions())
    path = output->get_extensions()->get<SignalPath>();

  if (path)
    path->reset();
  else
#endif
    get_output()->zero();
}


/***************************************************************************
 *
 *   Copyright (C) 2003 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SubFold.h"
#include "dsp/PhaseSeriesUnloader.h"

#include "Error.h"

using namespace std;

dsp::SubFold::SubFold ()
{
  built = false;
}

dsp::SubFold::~SubFold ()
{
}

dsp::Fold* dsp::SubFold::clone () const
{
  return new SubFold(*this);
}

//! Set the file unloader
void dsp::SubFold::set_unloader (dsp::PhaseSeriesUnloader* _unloader)
{
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

  while (more_data)
  {
    divider.set_bounds( get_input() );

    if (!divider.get_fractional_pulses())
      output->set_ndat_expected( divider.get_division_ndat() );

    more_data = divider.get_in_next ();

    if (divider.get_new_division() && output->get_integration_length())
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

    // flag that the current profile should be unloaded after the next fold
    if (!divider.get_end_reached())
      continue;

    if (verbose)
      cerr << "dsp::SubFold::transformation sub-integration completed" << endl;

    complete.send (output);

    if (!keep(output))
    {
      if (verbose)
	cerr << "dsp::SubFold::transformation discard sub-integration" << endl;
      output->zero();
      continue;
    }

    if (unloader)
    {
      if (verbose)
	cerr << "dsp::SubFold::transformation unload subint" << endl;

      unloader->unload(output);
    }

    output->zero();
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
  if (verbose)
    cerr << "dsp::SubFold::finish unload_partial" << endl;

  unload_partial ();

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
  if (!output->get_integration_length())
  {
    output->zero();
    return;
  }

  if (verbose)
    cerr << "dsp::SubFold::unload_partial to callback" << endl;
  
  partial.send (output);

  if (unloader)
  {
    if (verbose)
      cerr << "dsp::SubFold::unload_partial to unloader" << endl;

    unloader->partial (output);
  }

  output->zero();
}
catch (Error& error)
{
  throw error += "dsp::SubFold::finish";
}


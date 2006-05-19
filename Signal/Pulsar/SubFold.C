#include "dsp/SubFold.h"
#include "dsp/PhaseSeriesUnloader.h"

#include "polyco.h"
#include "Error.h"

#include <assert.h>

dsp::SubFold::SubFold ()
{
  built = false;
}

dsp::SubFold::~SubFold ()
{
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

void dsp::SubFold::prepare ()
{
  if (verbose)
    cerr << "dsp::SubFold::prepare call Fold::prepare" << endl;

  Fold::prepare ();

  // if unspecified, the first TimeSeries to be folded will define the
  // start time from which to begin cutting up the observation
  if (divider.get_start_time() == MJD::zero)
    divider.set_start_time (input->get_start_time());

  if (has_folding_polyco() && divider.get_turns())
    divider.set_polyco (get_folding_polyco());

  built = true;
}


void dsp::SubFold::transformation ()
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

  while (more_data) {

    divider.set_bounds( get_input() );

    more_data = divider.get_in_next ();

    if (divider.get_new_division() && output->get_integration_length()) {

      /* A new division has been started and there is still data in
	 the current integration.  This is a sign that the current
	 input comes from uncontiguous data, which can arise when
	 processing in parallel. */

      if (verbose)
	cerr << "dsp::SubFold::transformation"
	  " storing incomplete sub-integration" << endl;
	  
      partial.send (*output);
      output->zero();

    }

    if (!divider.get_is_valid())
      continue;

    Fold::transformation ();

    // flag that the current profile should be unloaded after the next fold
    if (!divider.get_end_reached())
      continue;

    if (verbose)
      cerr << "dsp::SubFold::transformation sub-integration completed" << endl;

    complete.send (*output);

    if (!keep(output)) {
      if (verbose)
	cerr << "dsp::SubFold::transformation discard sub-integration" << endl;
      output->zero();
      continue;
    }

    if (unloader) {
      if (verbose)
	cerr << ":dsp::SubFold::transformation unload subint" << endl;

      unloader->set_profiles(output);
      unloader->unload();
    }

    output->zero();


  }
}

/*! sets the Fold::idat_start and Fold::ndat_fold attributes */
void dsp::SubFold::set_limits (const Observation* input)
{
  idat_start = divider.get_idat_start ();
  ndat_fold = divider.get_ndat ();
}



#include "dsp/SubFold.h"
#include "dsp/PhaseSeriesUnloader.h"

#include "polyco.h"
#include "Error.h"

dsp::SubFold::SubFold ()
{
  subint_seconds = 0;
  subint_turns = 0;
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
void dsp::SubFold::set_start_time (MJD _start_time)
{
  if (start_time == _start_time)
    return;

  start_time = _start_time;
  start_phase = Phase::zero;
}

//! Set the interval over which to fold each sub-integration (in seconds)
void dsp::SubFold::set_subint_seconds (double _subint_seconds)
{
  if (subint_seconds == _subint_seconds)
    return;

  subint_seconds = _subint_seconds;
  subint_turns = 0;
}

//! Set the number of pulses to fold into each sub-integration
void dsp::SubFold::set_subint_turns (unsigned _subint_turns)
{
  if (subint_turns == _subint_turns)
    return;

  subint_turns = _subint_turns;
  subint_seconds = 0;
}


void dsp::SubFold::transformation ()
{
  if (verbose)
    cerr << "dsp::SubFold::transformation" << endl;

  if (subint_turns == 0 && subint_seconds == 0.0)
    throw Error (InvalidState, "dsp::SubFold::tranformation",
		 "sub-integration length not specified");


  // if unspecified, the first TimeSeries to be folded will define the
  // start time from which to begin cutting up the observation
  if (start_time == MJD::zero)
    set_start_time (input->get_start_time());

  // flag that the input TimeSeries contains data for another sub-integration
  bool more_data = true;

  while (more_data) {

    // flag that the current profile should be unloaded after the next fold
    bool subint_full = false;

    // SubFold::bound sets the Fold::idat_start and Fold::ndat_fold attributes
    bool fold = bound (more_data, subint_full);

    if (fold)
      Fold::transformation ();

    if (!subint_full)  {

      // the current sub-integration is not yet full ...

      if (more_data) {

	// ... however, the current input TimeSeries has data to
	// contribute.  This is a sign that the current input comes from
	// uncontiguous data, which can arise when processing in
	// parallel

	if (verbose)
	  cerr << "dsp::SubFold::transformation"
	    " storing incomplete sub-integration" << endl;

	subints.push_back (output);
	output = new PhaseSeries;

      }
      continue;
    }

    if (verbose)
      cerr << "dsp::SubFold::transformation sub-integration completed" << endl;

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
      output->zero();
    }

    else {
      if (verbose)
	cerr << "dsp::SubFold::transformation storing sub-integration" << endl;
      subints.push_back (output);
      output = new PhaseSeries;
    }

  }
}

void dsp::SubFold::set_limits (const Observation* input)
{
  /* Limits are set by the SubFold::bound method.  However,
   Fold::transformation calls this virtual method when it is called in
   SubFold::transformation.  Therefore, it is necessary to disable
   Fold::set_limits. */
}

/*! This method sets the idat_start and ndat_fold attributes of the
  Fold base class */
bool dsp::SubFold::bound (bool& more_data, bool& subint_full)
{
  more_data = false;
  subint_full = false;

  double sampling_rate = input->get_rate();
  uint64 input_ndat = input->get_ndat();

  MJD input_start = input->get_start_time();
  MJD input_end   = input->get_end_time();

  bool contains_data = (output->get_integration_length () > 0);

  //////////////////////////////////////////////////////////////////////////
  //
  // determine the MJD at which to start folding 
  //
  MJD fold_start;

  if (!contains_data) {

    // The current sub-integration contains no data.  Set new
    // boundaries and start folding with the first sample of the input
    // TimeSeries within the boundaries.

    if (lower == MJD::zero)
      set_boundaries (input_start);

    fold_start = std::max (lower, input_start);

    if (verbose)
      cerr << "dsp::SubFold::bound start new sub-integration at" 
	   << "\n        start = " << fold_start
	   << "\n subint start = " << lower
	   << "\n  input start = " << input_start
	   << endl;
  }
  else {

    // The current sub-integration contains data.  Check that the
    // input TimeSeries has data within the current boundaries.

    if (input_end < lower || input_start > upper) {

      if (verbose) cerr << "dsp::SubFold::bound"
		     " input not from this sub-integration" << endl;

      // This state: (more_data == true && subint_full == false)
      // indicates that the output PhaseSeries may be only partially
      // full.  The output should be set to an empty PhaseSeries and
      // this method should be called again.

      lower = MJD::zero;
      more_data = true;
      return false;

    }
    
    MJD output_end = output->get_end_time();

    fold_start = std::max (output_end, input_start);

    if (verbose)
      cerr << "dsp::SubFold::bound continue folding at" 
	   << "\n        start = " << fold_start
	   << "\n   output end = " << output_end
	   << "\n  input start = " << input_start
	   << endl;

  }


  //////////////////////////////////////////////////////////////////////////
  //
  // determine how far into the current input TimeSeries to start
  //
  MJD offset = fold_start - input_start;

  double start_sample = offset.in_seconds() * sampling_rate;
  idat_start = (uint64) rint (start_sample);

  if (verbose)
    cerr << "dsp::SubFold::bound start offset " << offset.in_seconds()*1e3
	 << " ms (" << idat_start << "pts)" << endl;
  
  if (idat_start >= input_ndat) {

    // The current data end before the start of the current
    // sub-integration

    if (verbose)
      cerr << "dsp::SubFold::bound data end before start of current subint=" 
	   << fold_start << endl;

    return false;

  }

  //////////////////////////////////////////////////////////////////////////
  //
  // determine how far into the current input TimeSeries to end
  //
  MJD fold_end = std::min (input_end, upper);

  if (verbose)
    cerr << "dsp::SubFold::bound end folding at "
         << "\n          end = " << fold_end
         << "\n   subint end = " << upper
         << "\n    input end = " << input_end
         << endl;

  offset = fold_end - input_start;

  double end_sample = offset.in_seconds() * sampling_rate;
  uint64 idat_end = (uint64) rint (end_sample);

  if (verbose)
    cerr << "dsp::SubFold::bound end offset " << offset.in_seconds()*1e3
	 << " ms (" << idat_end << "pts)" << endl;
  
  if (idat_end > input_ndat) {

    // this can happen owing to rounding in the above call to rint()

    if (verbose)
      cerr << "dsp::SubFold::bound fold"
	"\n   end_sample=rint(" << end_sample << ")=" << idat_end << 
	" > input ndat=" << input_ndat << endl;

    idat_end = input_ndat;

  }
  else if (idat_end < input_ndat) {

    // The current input TimeSeries extends more than the current
    // sub-integration.  The more_data flag indicates that the input
    // TimeSeries should be used again.

    if (verbose)
      cerr << "dsp::SubFold::bound input data ends "
           << (input_end-fold_end).in_seconds()*1e3 <<
        " ms after current sub-integration" << endl;

    more_data = true;

  }

  ndat_fold = idat_end - idat_start;

  //////////////////////////////////////////////////////////////////////////
  //
  // determine if the end of the current sub-integration has been reached
  //

  double samples_to_end = (upper - fold_end).in_seconds() * sampling_rate;

  if (verbose)
    cerr << "dsp::SubFold::bound " << samples_to_end << " samples to"
      " end of current sub-integration" << endl;

  if (samples_to_end < 0.5) {

    if (verbose)
      cerr << "dsp::SubFold::bound end of sub-integration" << endl;

    subint_full = true;
    set_boundaries (fold_end + 0.5/sampling_rate);

  }

  if (verbose) {
    double used = double(ndat_fold)/sampling_rate;
    double available = double(input_ndat)/sampling_rate;
    cerr << "dsp::SubFold::bound fold " << used*1e3 << "/"
	 << available*1e3 << " ms (" << ndat_fold << "/" << input_ndat
	 << " samples)" << endl;
  }

  return true;
}


void dsp::SubFold::set_boundaries (const MJD& input_start)
{
  if (start_time == MJD::zero)
    throw Error (InvalidState, "dsp::SubFold::set_boundaries",
		 "Observation start time not set");
 
  if (subint_turns && get_folding_polyco() && start_phase == Phase::zero) {

    // On the first call to set_boundaries, initialize to start at
    // Fold::reference_phase

    start_phase = get_folding_polyco()->phase(start_time);

    if (start_phase.fracturns() > reference_phase)
      ++ start_phase;

    start_phase = Phase (start_phase.intturns(), reference_phase);

    start_time = get_folding_polyco()->iphase (start_phase);

  }

  MJD fold_start = std::max (start_time, input_start);

  if (subint_turns && folding_period != 0) {
    // folding a specified number of turns at a constant period is
    // equivalent to folding a specified number of seconds
    subint_seconds = folding_period * double(subint_turns);
  }

  if (subint_seconds > 0) {

    // sub-integration length specified in seconds
    double seconds = (fold_start - start_time).in_seconds();

    // assumption: integer cast truncates
    uint64 subint = uint64 (seconds/subint_seconds);

    lower = start_time + double(subint) * seconds;
    upper = lower + seconds;

    return;
  }

  if (!subint_turns)
    throw Error (InvalidState, "dsp::SubFold::set_boundaries",
		 "sub-integration length not specified");

  // sub-integration length specified in turns

  if (!get_folding_polyco())
    throw Error (InvalidState, "dsp::SubFold::set_boundaries",
		 "sub-integration length specified in turns "
		 "but no folding period or polyco");

  if (verbose)
    cerr << "dsp::SubFold::set_boundaries using polynomial: "
      "avg. period=" << get_folding_polyco()->period(fold_start) << endl;

  Phase input_phase = get_folding_polyco()->phase (fold_start);

  double turns = (input_phase - start_phase).in_turns();

  // assumption: integer cast truncates
  uint64 subint = uint64 (turns/subint_turns);

  input_phase = start_phase + subint * subint_turns;
  lower = get_folding_polyco()->iphase (input_phase);
  
  input_phase += int(subint_turns);
  upper = get_folding_polyco()->iphase (input_phase);
}

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

      unloader->unload (output);
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

/*! This method sets the idat_start and ndat_fold attributes of the
  Fold base class */
bool dsp::SubFold::bound (bool& more_data, bool& subint_full)
{
  more_data = false;
  subint_full = false;

  double sampling_rate = input->get_rate();
  MJD input_start = input->get_start_time();
  MJD input_end   = input->get_end_time();

  bool contains_data = (output->get_integration_length () > 0);
  MJD output_end;

  if (contains_data)
    output_end = output->get_end_time();
  else
    output_end = lower;

  MJD fold_start;

  // some comparisons need not be so rigorous for the purposes of this
  // routine.  half the time resolution will do
  double oldMJDprecision = MJD::precision;
  MJD::precision = 0.5/sampling_rate;

  if (contains_data || lower != MJD::zero) {

    // check that the input TimeSeries has data within the current boundaries

    if (input_end < lower || input_start > upper) {
      if (verbose) cerr << "dsp::SubFold::bound"
		     " input not from this sub-integration" << endl;

      lower = MJD::zero;
      more_data = true;

      MJD::precision = oldMJDprecision;
      return false;
    }
    
    fold_start = std::max (output_end, input_start);

    if (verbose)
      cerr << "dsp::SubFold::bound continue folding at" 
	   << "\n        start = " << fold_start
	   << "\n   output end = " << output_end
	   << "\n  input start = " << input_start
	   << endl;

  }
  else {

    // the current subint contains no data and there are currently no
    // sub-integration boundaries set.  Set new boundaries and start
    // folding with the first sample of the input TimeSeries within the
    // boundaries.
    
    set_boundaries (input_start);

    fold_start = std::max (lower, input_start);

    if (verbose)
      cerr << "dsp::SubFold::bound start new sub-integration at" 
	   << "\n        start = " << fold_start
	   << "\n subint start = " << lower
	   << "\n  input start = " << input_start
	   << endl;
  }

  // determine the amount of data to be integrated, and how far into
  // the current input TimeSeries to start
  MJD offset = fold_start - input_start;
  double offset_samples = offset.in_seconds() * sampling_rate;
  idat_start = (uint64) rint (offset_samples);

  if (verbose)
    cerr << "dsp::SubFold::bound offset " << offset.in_seconds()*1e3
	 << " ms (" << idat_start << "pts)" << endl;
  
  MJD fold_end = std::min (input_end, upper);
  MJD fold_total = fold_end - fold_start;

  double fold_samples = fold_total.in_seconds() * sampling_rate;
  ndat_fold = (uint64) rint (fold_samples);

  if (verbose)
    cerr << "dsp::SubFold::bound fold " << fold_total.in_seconds()*1e3
	 << " ms (" << ndat_fold << "pts) until"
         << "\n          end = " << fold_end
         << "\n   subint end = " << upper
         << "\n    input end = " << input_end
         << endl;

  if (fold_total.in_seconds() < 0.0) {
    // the current data end before the start subint of interest
    if (verbose)
      cerr << "dsp::SubFold::bound data end before start of current subint=" 
	   << fold_start << endl;

    // return false - no error, but the caller should check the
    // integration length before using the subint
    MJD::precision = oldMJDprecision;
    return false;
  }

  if (idat_start + ndat_fold > input->get_ndat()) {
    // this can happen owing to rounding in the above two calls to rint()
    if (verbose)
      cerr << "dsp::SubFold::bound fold"
	"\n   offset=rint(" << offset_samples << ")=" << idat_start <<
	"\n +  total=rint(" << fold_samples << ")=" << ndat_fold <<
        "\n = " << idat_start+ndat_fold << 
	" > input ndat=" << input->get_ndat() << endl;
    ndat_fold = input->get_ndat() - idat_start;
  }

  double actual = double(ndat_fold)/sampling_rate;
  if (verbose)
    cerr << "dsp::SubFold::bound fold " << actual*1e3 << "/"
	 << (input_end - input_start).in_seconds()*1e3 << " ms ("
	 << ndat_fold << "/" << input->get_ndat() << " pts)" << endl;
  
  double samples_to_end = 
    (upper - fold_end).in_seconds() * sampling_rate;

  if (verbose)
    cerr << "dsp::SubFold::bound " << samples_to_end << " samples to"
      " end of current sub-integration" << endl;

  if (samples_to_end < 0.5)
    subint_full = true;

  if (fold_end < input_end) {
    // the current input TimeSeries extends more than the current
    // sub-integration: set the more_data flag and the boundaries for
    // the next sub-integration
    if (verbose)
      cerr << "dsp::SubFold::bound " << (input_end-fold_end).in_seconds()*1e3
	   << " ms after end of current sub-integration" << endl
	   << "dsp::SubFold::bound set bounds for next sub-integration"<<endl;

    set_boundaries (fold_end+MJD::precision);
    more_data = true;
  }

  MJD::precision = oldMJDprecision;
  return true;
}


void dsp::SubFold::set_boundaries (const MJD& input_start)
{
  if (start_time == MJD::zero)
    throw Error (InvalidState, "dsp::SubFold::set_boundaries",
		 "Observation start time not set");

  if (subint_turns && folding_polyco && start_phase == Phase::zero) {
    // round up to start at zero phase
    start_phase = folding_polyco->phase(start_time).Ceil();
    start_time = folding_polyco->iphase (start_phase);
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

  if (!folding_polyco)
    throw Error (InvalidState, "dsp::SubFold::set_boundaries",
		 "sub-integration length specified in turns "
		 "but no folding period or polyco");

  if (verbose)
    cerr << "dsp::SubFold::set_boundaries using polynomial: "
      "avg. period=" << folding_polyco->period(fold_start) << endl;

  Phase input_phase = folding_polyco->phase (fold_start);

  double turns = (input_phase - start_phase).in_turns();
  // assumption: integer cast truncates
  uint64 subint = uint64 (turns/subint_turns);

  input_phase = start_phase + subint * subint_turns;
  lower = folding_polyco->iphase (input_phase);
  
  input_phase += double(subint_turns);
  upper = folding_polyco->iphase (input_phase);
}

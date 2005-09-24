#include "dsp/TimeDivide.h"
#include "dsp/Observation.h"
#include "dsp/Operation.h"

#include "Error.h"

#include <assert.h>

dsp::TimeDivide::TimeDivide ()
{
  division_seconds = 0;
  division_turns = 0;

  in_current = false;
  in_next = false;
  end_reached = false;
  contiguous = false;
}

dsp::TimeDivide::~TimeDivide ()
{
}

void dsp::TimeDivide::set_start_time (MJD _start_time)
{
  if (start_time == _start_time)
    return;

  start_time  = _start_time;
  start_phase = Phase::zero;
}

void dsp::TimeDivide::set_seconds (double seconds)
{
  if (division_seconds == seconds)
    return;

  division_seconds = seconds;
  division_turns = 0;
}

void dsp::TimeDivide::set_turns (double turns)
{
  if (division_turns == turns)
    return;

  division_turns = turns;
  division_seconds = 0;
}

void dsp::TimeDivide::set_polyco (const polyco* _poly)
{
  if (poly && poly == _poly)
    return;

  poly = _poly;
  division_seconds = 0;
}

//! Set the reference phase (phase of bin zero)
void dsp::TimeDivide::set_reference_phase (double phase)
{
  // ensure that phase runs from 0 to 1
  phase -= floor (phase);
  reference_phase = phase;
}

void dsp::TimeDivide::set_bounds (const Observation* input)
{
  in_current = false;
  in_next = false;
  end_reached = false;

  double sampling_rate = input->get_rate();
  uint64 input_ndat = input->get_ndat();

  MJD input_start = input->get_start_time();
  MJD input_end   = input->get_end_time();

  //////////////////////////////////////////////////////////////////////////
  //
  // determine the MJD at which to start
  //
  MJD divide_start;

  if (!contiguous) {

    /* Set new boundaries and start the division with the first sample
       of the input Observation within the boundaries. */

    set_boundaries (input_start);

    divide_start = std::max (lower, input_start);

    if (Operation::verbose)
      cerr << "dsp::TimeDivide::bound start new division at" 
	   << "\n          start = " << divide_start
	   << "\n division start = " << lower
	   << "\n    input start = " << input_start
	   << endl;

  }

  else {
   
    divide_start = std::max (current_end, input_start);

    if (Operation::verbose)
      cerr << "dsp::TimeDivide::bound continue at" 
	   << "\n        start = " << divide_start
	   << "\n  current end = " << current_end
	   << "\n  input start = " << input_start
	   << endl;

  }

  // Check that the Observation is within the current boundaries.

  if (input_end < lower || input_start > upper) {

    if (Operation::verbose) cerr << "dsp::TimeDivide::bound"
	     " input not from this sub-integration" << endl;

    /*  
	This state (in_next == true && end_reached == false) indicates
	that the output PhaseSeries may be only partially full.  The
	output should be reset and this method should be called again.
    */

    in_next = true;
    contiguous = false;

    return;
  }

  // the current observation is within the current boundaries
  contiguous = true;

  //////////////////////////////////////////////////////////////////////////
  //
  // determine how far into the current Observation to start
  //
  MJD offset = divide_start - input_start;

  double start_sample = offset.in_seconds() * sampling_rate;

  //cerr << "start_sample=" << start_sample << endl;
  start_sample = rint (start_sample);
  //cerr << "rint(start_sample)=" << start_sample << endl;

  assert (start_sample >= 0);

  idat_start = (uint64) start_sample;

  if (Operation::verbose)
    cerr << "dsp::TimeDivide::bound start offset " << offset.in_seconds()*1e3
	 << " ms (" << idat_start << "pts)" << endl;
  
  if (idat_start >= input_ndat) {

    // The current data end before the start of the current
    // sub-integration

    if (Operation::verbose)
      cerr << "dsp::TimeDivide::bound data end before start of"
	" current division=" << divide_start << endl;

    return;

  }

  //////////////////////////////////////////////////////////////////////////
  //
  // determine how far into the current input TimeSeries to end
  //
  MJD divide_end = std::min (input_end, upper);

  if (Operation::verbose)
    cerr << "dsp::TimeDivide::bound end division at "
         << "\n           end = " << divide_end
         << "\n  division end = " << upper
         << "\n     input end = " << input_end
         << endl;

  offset = divide_end - input_start;

  double end_sample = offset.in_seconds() * sampling_rate;

  //cerr << "end_sample=" << end_sample << endl;
  end_sample = rint (end_sample);
  //cerr << "rint(end_sample)=" << end_sample << endl;

  assert (end_sample >= 0);

  uint64 idat_end = (uint64) end_sample;

  if (Operation::verbose)
    cerr << "dsp::TimeDivide::bound end offset " << offset.in_seconds()*1e3
	 << " ms (" << idat_end << "pts)" << endl;
  
  if (idat_end > input_ndat) {

    // this can happen owing to rounding in the above call to rint()

    if (Operation::verbose)
      cerr << "dsp::TimeDivide::bound division"
	"\n   end_sample=rint(" << end_sample << ")=" << idat_end << 
	" > input ndat=" << input_ndat << endl;

    idat_end = input_ndat;

  }
  else if (idat_end < input_ndat) {

    // The current input TimeSeries extends more than the current
    // sub-integration.  The in_next flag indicates that the input
    // TimeSeries should be used again.

    if (Operation::verbose)
      cerr << "dsp::TimeDivide::bound input data ends "
           << (input_end-divide_end).in_seconds()*1e3 <<
        " ms after current sub-integration" << endl;

    in_next = true;

  }

  ndat = idat_end - idat_start;

  //////////////////////////////////////////////////////////////////////////
  //
  // determine if the end of the current sub-integration has been reached
  //

  double samples_to_end = (upper - divide_end).in_seconds() * sampling_rate;

  if (Operation::verbose)
    cerr << "dsp::TimeDivide::bound " << samples_to_end << " samples to"
      " end of current sub-integration" << endl;

  if (samples_to_end < 0.5) {

    if (Operation::verbose)
      cerr << "dsp::TimeDivide::bound end of sub-integration" << endl;

    end_reached = true;
    set_boundaries (divide_end + 0.5/sampling_rate);

  }

  if (Operation::verbose) {
    double used = double(ndat)/sampling_rate;
    double available = double(input_ndat)/sampling_rate;
    cerr << "dsp::TimeDivide::bound division " << used*1e3 << "/"
	 << available*1e3 << " ms (" << ndat << "/" << input_ndat
	 << " samples)" << endl;
  }

  current_end = input_start + idat_end;
  in_current = true;
}


void dsp::TimeDivide::set_boundaries (const MJD& input_start)
{
  if (start_time == MJD::zero)
    throw Error (InvalidState, "dsp::TimeDivide::set_boundaries",
		 "Observation start time not set");
 
  if (division_turns && poly && start_phase == Phase::zero) {

    /* On the first call to set_boundaries, initialize to start at
       reference_phase */

    if (Operation::verbose)
      cerr << "dsp::TimeDivide::set_boundaries first call\n\treference_phase="
           << reference_phase << " start_time=" << start_time << endl;

    start_phase = poly->phase(start_time);

    if (start_phase.fracturns() > reference_phase)
      ++ start_phase;

    start_phase = Phase (start_phase.intturns(), reference_phase);

    start_time = poly->iphase (start_phase);

    if (Operation::verbose)
      cerr << "dsp::TimeDivide::set_boundaries first call\n\tstart_phase="
           << start_phase << " start_time=" << start_time << endl;

  }

  MJD divide_start = std::max (start_time, input_start);

  if (division_seconds > 0) {

    // sub-integration length specified in seconds
    double seconds = (divide_start - start_time).in_seconds();

    // assumption: integer cast truncates
    uint64 division = uint64 (seconds/division_seconds);

    lower = start_time + double(division) * seconds;
    upper = lower + seconds;

    return;
  }

  if (!division_turns)
    throw Error (InvalidState, "dsp::TimeDivide::set_boundaries",
		 "division length not specified");

  // sub-integration length specified in turns

  if (!poly)
    throw Error (InvalidState, "dsp::TimeDivide::set_boundaries",
		 "division length specified in turns "
		 "but no folding polyco");

  if (Operation::verbose)
    cerr << "dsp::TimeDivide::set_boundaries using polynomial: "
      "avg. period=" << poly->period(divide_start) << endl;

  Phase input_phase = poly->phase (divide_start);

  double turns = (input_phase - start_phase).in_turns();

  // assumption: integer cast truncates
  uint64 division = uint64 (turns/division_turns);

  input_phase = start_phase + division * division_turns;
  lower = poly->iphase (input_phase);
  
  input_phase += int(division_turns);
  upper = poly->iphase (input_phase);
}

/***************************************************************************
 *
 *   Copyright (C) 2005 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/TimeDivide.h"
#include "dsp/Observation.h"
#include "dsp/Operation.h"

#include "Error.h"

#include <assert.h>

using namespace std;

// #define _DEBUG 1

dsp::TimeDivide::TimeDivide ()
{
  division_seconds = 0;
  division_turns = 0;
  phase_bin = 0;

  is_valid = false;
  in_next = false;
  end_reached = false;
  new_division = false;
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
  is_valid = false;
}

void dsp::TimeDivide::set_seconds (double seconds)
{
  if (division_seconds == seconds)
    return;

  division_seconds = seconds;
  if (seconds)
    division_turns = 0;
}

void dsp::TimeDivide::set_turns (double turns)
{
  if (division_turns == turns)
    return;

  division_turns = turns;
  if (turns)
    division_seconds = 0;
}

void dsp::TimeDivide::set_predictor (const Pulsar::Predictor* _poly)
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
  double sampling_rate = input->get_rate();
  uint64 input_ndat = input->get_ndat();

  MJD input_start = input->get_start_time();
  MJD input_end   = input->get_end_time();

  //////////////////////////////////////////////////////////////////////////
  //
  // determine the MJD at which to start
  //
  MJD divide_start = input_start;

  if (is_valid)
  {
    divide_start = std::max (current_end, input_start);

    if (Operation::verbose)
      cerr << "dsp::TimeDivide::bound continue at" 
	   << "\n        start = " << divide_start
	   << "\n  current end = " << current_end
	   << "\n  input start = " << input_start
	   << endl;
  }

  new_division = false;
  end_reached = false;
  in_next = false;

  if (input_end < lower || divide_start+0.6/sampling_rate > upper) {

    /*
      This state occurs when either:
      1) this method is first called (no boundaries set)
      2) the 
    */

    if (Operation::verbose)  {
      cerr << "dsp::TimeDivide::bound start new division" << endl;
      if (input_end < lower)
        cerr << "      input end = " << input_end << " precedes\n"
                " division start = " << lower << endl;
       else
        cerr << "          start = " << divide_start << " is after\n"
                "   division end = " << upper << endl;
    }

    new_division = true;

    set_boundaries (divide_start + 0.5/sampling_rate);

  }

  divide_start = std::max (lower, divide_start);

  if (Operation::verbose)
    cerr << "dsp::TimeDivide::bound start division at" 
	 << "\n          start = " << divide_start
	 << "\n division start = " << lower
	 << "\n    input start = " << input_start
	 << endl;

  //////////////////////////////////////////////////////////////////////////
  //
  // determine how far into the current Observation to start
  //
  MJD offset = divide_start - input_start;

  double start_sample = offset.in_seconds() * sampling_rate;

  // cerr << "start_sample=" << start_sample << endl;
  start_sample = rint (start_sample);
  // cerr << "rint(start_sample)=" << start_sample << endl;

  assert (start_sample >= 0);

  idat_start = (uint64) start_sample;

  if (Operation::verbose)
    cerr << "dsp::TimeDivide::bound start offset " << offset.in_seconds()*1e3
	 << " ms (" << idat_start << "pts)" << endl;

  if (idat_start >= input_ndat) {

    // The current data end before the start of the current division

    if (Operation::verbose)
      cerr << "dsp::TimeDivide::bound input ends before division starts"<<endl;

    is_valid = false;
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

  // cerr << "end_sample=" << end_sample << endl;
  end_sample = rint (end_sample);
  // cerr << "rint(end_sample)=" << end_sample << endl;

  assert (end_sample >= 0);

  uint64 idat_end = (uint64) end_sample;

  if (Operation::verbose)
    cerr << "dsp::TimeDivide::bound end offset " << offset.in_seconds()*1e3
	 << " ms (" << idat_end << "pts)" << endl;
 
  if (idat_end <= idat_start)
    throw Error (InvalidState, "dsp::TimeDivide::bound",
		 "idat_end="UI64" <= idat_start="UI64, idat_end, idat_start);

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
    // division.  The in_next flag indicates that the input
    // TimeSeries should be used again.

    if (Operation::verbose)
      cerr << "dsp::TimeDivide::bound input data ends "
           << (input_end-divide_end).in_seconds()*1e3 <<
        " ms after current division" << endl;

    in_next = true;

  }

  ndat = idat_end - idat_start;

  //////////////////////////////////////////////////////////////////////////
  //
  // determine if the end of the current division has been reached
  //

  double samples_to_end = (upper - divide_end).in_seconds() * sampling_rate;

  if (Operation::verbose)
    cerr << "dsp::TimeDivide::bound " << samples_to_end << 
      " samples to end of current division" << endl;

  if (samples_to_end < 0.5) {
    if (Operation::verbose)
      cerr << "dsp::TimeDivide::bound end of division" << endl;

    end_reached = true;
  }

  if (Operation::verbose) {
    double start = 1e3*idat_start/sampling_rate;
    double used = 1e3*ndat/sampling_rate;
    double available = 1e3*input_ndat/sampling_rate;
    cerr << "dsp::TimeDivide::bound using "
	 << used << "/" << available << " from " << start << " ms\n  (" 
	 << ndat << "/" << input_ndat << " from " << idat_start << " to "
	 << idat_start + ndat - 1 << " inclusive.)" << endl;
  }

  is_valid = true;
  current_end = input_start + idat_end/sampling_rate;
}

void dsp::TimeDivide::discard_bounds (const Observation* input)
{
  double sampling_rate = input->get_rate();
  current_end -= ndat/sampling_rate;
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

    if (division_turns < 1.0) {

#ifdef _DEBUG
      cerr << "START PHASE=" << start_phase << endl;
#endif

      /* Find X, where:
	 X = required start_phase
	 y = current start_phase
	 D = division_turns
	 R = reference_phase

	 X > y
	 X = R + N*D
	 X < y + D
      */

      // X - R
      double XminusR = start_phase.fracturns() - reference_phase;

      // ensure that N > 0
      if (start_phase.fracturns() < reference_phase) {
	XminusR += 1.0;
	-- start_phase;
      }

#ifdef _DEBUG
      cerr << "OFFSET FROM REFERENCE=" << XminusR << endl;
#endif

      // N = (X-R)/D
      unsigned N = (unsigned) ceil (XminusR / division_turns);

#ifdef _DEBUG
      cerr << "NEXT PHASE BIN=" << N << endl;
#endif

      // X = R + N*D
      double X = reference_phase + N * division_turns;

#ifdef _DEBUG
      cerr << "START PHASE OF NEXT PHASE BIN=" << X << endl;
#endif

      start_phase = Phase (start_phase.intturns(), X);

#ifdef _DEBUG
      cerr << "START PHASE=" << start_phase << endl;
#endif

    }
    else {

      if (start_phase.fracturns() > reference_phase)
	++ start_phase;

      start_phase = Phase (start_phase.intturns(), reference_phase);

    }

    start_time = poly->iphase (start_phase);

    if (Operation::verbose)
      cerr << "dsp::TimeDivide::set_boundaries first call\n\tstart_phase="
           << start_phase << " start_time=" << start_time << endl;

  }

  MJD divide_start = std::max (start_time, input_start);

  if (division_seconds > 0) {

    // division length specified in seconds
    double seconds = (divide_start - start_time).in_seconds();

    // assumption: integer cast truncates
    division = uint64 (seconds/division_seconds);

#ifdef _DEBUG
    cerr << " divide_start=" << divide_start.printdays(13)
	 << " start_time=" << start_time.printdays(13)
	 << "\n seconds=" << seconds 
	 << " division=" << division << endl;
#endif

    lower = start_time + double(division) * division_seconds;
    upper = lower + division_seconds;

    return;
  }

  if (!division_turns)
    throw Error (InvalidState, "dsp::TimeDivide::set_boundaries",
		 "division length not specified");

  // division length specified in turns

  if (!poly)
    throw Error (InvalidState, "dsp::TimeDivide::set_boundaries",
		 "division length specified in turns "
		 "but no folding Pulsar::Predictor");

  if (Operation::verbose)
    cerr << "dsp::TimeDivide::set_boundaries using polynomial:\n"
      "  avg. period=" << 1.0/poly->frequency(divide_start) << endl;

  Phase input_phase = poly->phase (divide_start);

  double turns = (input_phase - start_phase).in_turns();

  // assumption: integer cast truncates
  division = uint64 (turns/division_turns);

  input_phase = start_phase + division * division_turns;
  lower = poly->iphase (input_phase);
  
  if (division_turns < 1.0) {
    Phase profile_phase = input_phase - reference_phase + 0.5 * division_turns;
    phase_bin = unsigned( profile_phase.fracturns() / division_turns );

#ifdef _DEBUG
    cerr << "division=" << division << " phase=" << profile_phase
	 << " bin=" << phase_bin << endl;
#endif

  }

  input_phase += division_turns;
  upper = poly->iphase (input_phase);

}

uint64 dsp::TimeDivide::get_division (const MJD& epoch)
{
  set_boundaries( epoch );
  return division;
}

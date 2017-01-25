//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2005 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/Pulsar/dsp/TimeDivide.h

#ifndef __baseband_dsp_TimeDivide_h
#define __baseband_dsp_TimeDivide_h

#include "environ.h"
#include "Pulsar/Predictor.h"
#include "OwnStream.h"

namespace dsp {

  class Observation;

  //! Calculates the boundaries of a division of time
  class TimeDivide : public OwnStream {

  public:
    
    //! Constructor
    TimeDivide ();

    //! Destructor
    ~TimeDivide ();

    /** @name Initialization Methods
     *  Use these methods to set up the division of time. */
    //@{

    //! Set the start time at which to begin dividing
    void set_start_time (MJD start_time);

    //! Get the start time at which to begin dividing
    MJD get_start_time () const { return start_time; }

    //! Set the number of seconds in each division
    void set_seconds (double division_seconds);

    //! Get the number of seconds in each division
    double get_seconds () const { return division_seconds; }

    //! Set the reference epoch (start time of first division)
    void set_reference_epoch (const MJD& epoch) { reference_epoch = epoch; }

    //! Set the reference epoch (start time of first division)
    MJD get_reference_epoch () const { return reference_epoch; }

    //! Set the number of turns in each division
    void set_turns (double division_turns);

    //! Get the number of turns in each division
    double get_turns () const { return division_turns; }

    //! Set the Pulsar::Predictor used to determine pulse phase
    void set_predictor (const Pulsar::Predictor*);

    //! Get the Pulsar::Predictor used to determine pulse phase
    const Pulsar::Predictor* get_predictor () const { return poly; }

    //! Set the folding period used to determine pulse phase
    void set_period (double);

    //! Set the folding period used to determine pulse phase
    double get_period () const { return period; }

    //! Set the reference phase (phase of bin zero)
    void set_reference_phase (double phase);

    //! Get the reference phase (phase of bin zero)
    double get_reference_phase () const { return reference_phase; }

    //! Fold the fractional pulses at the start and end of data
    void set_fractional_pulses (bool);
    bool get_fractional_pulses () const;

    //@}

    /** @name Operation Methods
     *  Use these methods to perform the division of time. */
    //@{

    //! Calculate the boundaries of the current division
    void set_bounds (const Observation*);

    //! Call this method if, after calling set_bounds, the data were not used
    void discard_bounds (const Observation*);
    
    //! Return true if the last bound Observation covers the current division
    bool get_is_valid () const { return is_valid; }

    //! Return true if the last bound Observation covers a new division
    bool get_new_division () const { return new_division; }

    //! Return true if the last bound Observation covers the next division
    bool get_in_next () const { return in_next; }

    //! Return true if the end of the current division has been reached
    bool get_end_reached () const { return end_reached; }

    //! Get the first time sample from observation in the current division
    uint64_t get_idat_start () const { return idat_start; }

    //! Get the number of time samples from observation in the current division
    uint64_t get_ndat () const { return ndat; }

    //! Get the total number of time samples in the current division
    uint64_t get_division_ndat () const { return division_ndat; }

    //! Get the phase bin of the current division (turns < 1)
    unsigned get_phase_bin () const { return phase_bin; }

    //! Get the current division
    uint64_t get_division () const { return division; }

    //! Get the division associated with the specified epoch
    uint64_t get_division (const MJD& epoch);

    //@}

  protected:

    //! The start time from which to begin dividing time
    MJD start_time;

    //! The corresponding phase at which to begin dividing time
    Pulsar::Phase start_phase;

    //! Number of seconds in each division
    double division_seconds;

    //! Reference epoch at start of the first division
    MJD reference_epoch;

    //! Number of turns in each division
    double division_turns;

    //! The reference phase
    double reference_phase;

    //! Include the fractional pulses at the start and end of data
    bool fractional_pulses;

    //! Round division boundaries to integer numbers of division_seconds
    bool integer_division_seconds_boundaries;

    //! The period used to determine pulse phase
    double period;

    //! The Pulsar::Predictor used to determine pulse phase
    Reference::To<const Pulsar::Predictor> poly;

    //! Calculates the boundaries of the sub-integration containing time
    void set_boundaries (const MJD& time);

    //! If observation is set, round boundaries to integer samples
    void set_boundaries (const MJD& mjd1, const MJD& mjd2);

  private:

    //! The start of the current division
    MJD lower;

    //! The end of the current division
    MJD upper;

    //! The end of the last Observation bound in this division
    MJD current_end;

    //! The index of the current division
    uint64_t division;

    //! Flag set when the divided time range is within the current division
    bool is_valid;

    //! Flag set when the divided time range extends past the current division
    bool in_next;

    //! Flag set when the end of the current division is reached
    bool end_reached;

    //! Flag set when the current observation appears contiguous
    bool new_division;

    //! The observation from which to compute time samples
    mutable const Observation* observation;

    //! The first time sample from observation in the current division
    uint64_t idat_start;

    //! The number of time samples from observation in the current division
    uint64_t ndat;

    //! The total number of time samples in the current division
    uint64_t division_ndat;

    //! The phase bin of the current division (division_turns < 1)
    unsigned phase_bin;

  };

}

#endif // !defined(__SubFold_h)

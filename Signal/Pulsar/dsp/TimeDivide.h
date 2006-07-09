//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2005 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/TimeDivide.h,v $
   $Revision: 1.4 $
   $Date: 2006/07/09 13:27:14 $
   $Author: wvanstra $ */

#ifndef __baseband_dsp_TimeDivide_h
#define __baseband_dsp_TimeDivide_h

#include "poly.h"
#include "Reference.h"

namespace dsp {

  class Observation;

  //! Calculates the boundaries of a division of time
  class TimeDivide {

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

    //! Set the number of turns in each division
    void set_turns (double division_turns);

    //! Get the number of turns in each division
    double get_turns () const { return division_turns; }

    //! Set the polyco used to determine pulse phase
    void set_polyco (const polyco*);

    //! Get the polyco used to determine pulse phase
    const polyco* get_polyco () const { return poly; }

    //! Set the reference phase (phase of bin zero)
    void set_reference_phase (double phase);

    //! Get the reference phase (phase of bin zero)
    double get_reference_phase () const { return reference_phase; }

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

    //! Get the first time sample in the current division
    uint64 get_idat_start () const { return idat_start; }

    //! Get the number of time samples in the current division
    uint64 get_ndat () const { return ndat; }

    //! Get the phase bin of the current division (turns < 1)
    unsigned get_phase_bin () const { return phase_bin; }

    //@}

  protected:

    //! The start time from which to begin dividing time
    MJD start_time;

    //! The corresponding phase at which to begin dividing time
    Phase start_phase;

    //! Number of seconds in each division
    double division_seconds;

    //! Number of turns in each division
    double division_turns;

    //! The polyco used to determine pulse phase
    Reference::To<const polyco> poly;

    //! The reference phase
    double reference_phase;

    //! Calculates the boundaries of the sub-integration containing time
    void set_boundaries (const MJD& time);

  private:

    //! The start of the current division
    MJD lower;

    //! The end of the current division
    MJD upper;

    //! The end of the last Observation bound in this division
    MJD current_end;


    //! Flag set when the divided time range is within the current division
    bool is_valid;

    //! Flag set when the divided time range extends past the current division
    bool in_next;

    //! Flag set when the end of the current division is reached
    bool end_reached;

    //! Flag set when the current observation appears contiguous
    bool new_division;


    //! The first time sample in the current division
    uint64 idat_start;

    //! The number of time samples in the current division
    uint64 ndat;

    //! The phase bin of the current division (division_turns < 1)
    unsigned phase_bin;

  };

}

#endif // !defined(__SubFold_h)

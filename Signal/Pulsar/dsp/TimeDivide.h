//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/TimeDivide.h,v $
   $Revision: 1.1 $
   $Date: 2005/09/24 12:56:00 $
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
    
    //! Return true if the last bound Observation covers the current division
    bool get_in_current () const { return in_current; }

    //! Return true if the last bound Observation covers the next division
    bool get_in_next () const { return in_next; }

    //! Return true if the end of the current division has been reached
    bool get_end_reached () const { return end_reached; }

    //! Get the first time sample in the current division
    uint64 get_idat_start () const { return idat_start; }

    //! Get the number of time samples in the current division
    uint64 get_ndat () const { return ndat; }

    //@}

  protected:

    //! The start time from which to begin counting sub-integrations
    MJD start_time;

    //! Number of seconds in each division
    double division_seconds;

    //! Number of turns in each division
    double division_turns;

    //! The polyco used to determine pulse phase
    Reference::To<const polyco> poly;

    //! The reference phase
    double reference_phase;

    //! Calculates the boundaries within which to fold the input TimeSeries
    bool bound (bool& more_data, bool& end_reached);

    //! Calculates the boundaries of the sub-integration containing time
    void set_boundaries (const MJD& time);

  private:

    //! The start of the current division
    MJD lower;

    //! The end of the current division
    MJD upper;

    //! The end of the last Observation bound in this division
    MJD current_end;

    //! The phase at which the current division starts
    Phase start_phase;

    //! Flag set when the divided time range is within the current division
    bool in_current;

    //! Flag set when the divided time range extends past the current division
    bool in_next;

    //! Flag set when the end of the current division is reached
    bool end_reached;

    //! Flag set when the current observation appears contiguous
    bool contiguous;

    //! The first time sample in the current division
    uint64 idat_start;

    //! The number of time samples in the current division
    uint64 ndat;
  };

}

#endif // !defined(__SubFold_h)

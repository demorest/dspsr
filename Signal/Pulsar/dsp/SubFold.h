//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/SubFold.h,v $
   $Revision: 1.1 $
   $Date: 2003/01/31 13:18:48 $
   $Author: wvanstra $ */

#ifndef __SubFold_h
#define __SubFold_h

#include "dsp/Fold.h"
#include "Phase.h"

namespace dsp {

  //! Fold data into sub-integrations
  /*! The SubFold class is useful for producing multiple sub-integrations
    from a single observation.  Given a polyco and the number of pulses
    to integrate, this class can be used to produce single pulse profiles.

    If the file unloading flag is set to false (see SubFold::set_unload)
    the SubFold class will store folded sub-integrations in the subints 
    vector.  If it is true, the SubFold class will call SubFold::unload,
    a virtual method that must be defined by a derived class that knows
    about archive file formats. */

  class SubFold : public Fold {

  public:
    
    //! Constructor
    SubFold ();
    
    //! Destructor
    ~SubFold ();
    
    //! Set the file unloading flag
    void set_unload (bool _unload = true) { unload_data = _unload; }

    //! Get the file unloading flag
    bool get_unload () const { return unload_data; }

    //! Set the start time from which to begin counting sub-integrations
    void set_start_time (MJD start_time);

    //! Get the start time from which to begin counting sub-integrations
    MJD get_start_time () const { return start_time; }

    //! Set the number of seconds to fold into each sub-integration
    void set_subint_seconds (double subint_seconds);

    //! Get the number of seconds to fold into each sub-integration
    double get_subint_seconds () const { return subint_seconds; }

    //! Set the number of turns to fold into each sub-integration
    void set_subint_turns (unsigned subint_turns);

    //! Get the number of turns to fold into each sub-integration
    unsigned get_subint_turns () const { return subint_turns; }

    //! Store the folded profile somewhere safe
    virtual void unload (PhaseSeries* data);

    //! Decide wether or not to keep the folded profile
    virtual bool keep (PhaseSeries* data) { return true; }

    //! Prepare to fold with the current attributes
    void prepare ();

  protected:

    //! Folds the TimeSeries data into one or more sub-integrations
    virtual void transformation ();

    //! If unload == false, sub-integrations are stored here
    vector< Reference::To<PhaseSeries> > subints;

    //! File unloading flag
    bool unload_data;

    //! The start time from which to begin counting sub-integrations
    MJD start_time;

    //! Interval over which to fold each sub-integration (in seconds)
    double subint_seconds;

    //! Number of turns to fold into each sub-integration
    unsigned subint_turns;

    //! Calculates the boundaries within which to fold the input TimeSeries
    bool bound (bool& more_data, bool& subint_full);

    //! Calculates the boundaries of the sub-integration containing time
    void set_boundaries (const MJD& time);

  private:

    //! The start of the current sub-integration
    MJD lower;

    //! The end of the current sub-integration
    MJD upper;

    //! The phase at which folding starts
    Phase start_phase;

  };

}

#endif // !defined(__SubFold_h)

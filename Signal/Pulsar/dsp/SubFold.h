//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2003 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/SubFold.h,v $
   $Revision: 1.13 $
   $Date: 2007/11/24 10:52:11 $
   $Author: straten $ */

#ifndef __SubFold_h
#define __SubFold_h

#include "dsp/Fold.h"
#include "dsp/TimeDivide.h"

#include "Phase.h"
#include "Callback.h"

namespace dsp {

  //! Fold data into sub-integrations
  /*! The SubFold class is useful for producing multiple sub-integrations
    from a single observation.  Given a Predictor and the number of pulses
    to integrate, this class can be used to produce single pulse profiles.

    If no PhaseSeriesUnloader is set (see SubFold::set_unloader),
    the SubFold class will store folded sub-integrations in the subints 
    vector.  If set, the SubFold class will call unloader->unload (output)
  */

  class PhaseSeriesUnloader;

  class SubFold : public Fold {

  public:
    
    //! Constructor
    SubFold ();
    
    //! Destructor
    ~SubFold ();
    
    //! Create a clonse
    Fold* clone () const;

    /** @name PhaseSeries events
     *  The attached callback methods should be of the form:
     *
     *  void method (const PhaseSeries& data);
     */
    //@{

    //! Attach methods to receive completed PhaseSeries instances
    Callback<PhaseSeries*> complete;

    //! Attach methods to receive partially completed PhaseSeries instances
    Callback<PhaseSeries*> partial;

    //@}

    //! Set the start time from which to begin counting sub-integrations
    void set_start_time (const MJD& start_time);

    //! Get the start time from which to begin counting sub-integrations
    MJD get_start_time () const { return divider.get_start_time(); }

    //! Set the number of seconds to fold into each sub-integration
    void set_subint_seconds (double subint_seconds);

    //! Get the number of seconds to fold into each sub-integration
    double get_subint_seconds () const { return divider.get_seconds(); }

    //! Set the number of turns to fold into each sub-integration
    void set_subint_turns (unsigned subint_turns);

    //! Get the number of turns to fold into each sub-integration
    unsigned get_subint_turns () const { return unsigned(divider.get_turns());}

    /** @name deprecated methods 
     *  Use of these methods is deprecated in favour of attaching
     *  callback methods to the completed event. */
    //@{

    //! Set the file unloader
    void set_unloader (PhaseSeriesUnloader* unloader);

    //! Get the file unloader
    PhaseSeriesUnloader* get_unloader () const;

    //! Decide wether or not to keep the folded profile
    virtual bool keep (PhaseSeries* data) { return true; }

    //@}

    //! Access to the divider
    const TimeDivide* get_divider () const { return &divider; }

  protected:

    //! Folds the TimeSeries data into one or more sub-integrations
    virtual void transformation ();

    //! Set the idat_start and ndat_fold attributes
    virtual void set_limits (const Observation* input);

    //! File unloading flag
    Reference::To<PhaseSeriesUnloader> unloader;
    
    //! The time divider
    TimeDivide divider;

    //! Initialize the time divider
    void prepare ();

    //! Flag set when time divider is initialized
    bool built;

  };

}

#endif // !defined(__SubFold_h)

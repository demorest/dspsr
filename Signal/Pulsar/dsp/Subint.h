//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2010 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __Subint_h
#define __Subint_h

#include "dsp/TimeDivide.h"
#include "dsp/PhaseSeries.h"

#include "Phase.h"
#include "Callback.h"

namespace dsp {

  class PhaseSeriesUnloader;

  //! Unload PhaseSeries data into sub-integrations
  /*! The Subint class is useful for producing multiple sub-integrations
    from a single observation.  Given a Predictor and the number of pulses
    to integrate, this class can be used to produce single pulse profiles.

    This class has been reimplemented from SubFold as a template to allow
    the subintegration division code to apply to any base class that
    produces PhaseSeries output.
  */

  template <class Op>
  class Subint : public Op {

  public:
    
    //! Constructor
    Subint ();
    
    //! Destructor
    ~Subint ();
    
    //! Create a clonse
    Op* clone () const;

    //! Emit any unfinished profiles
    void finish ();

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
    void set_start_time (const MJD& start_time)
      { divider.set_start_time (start_time); }

    //! Get the start time from which to begin counting sub-integrations
    MJD get_start_time () const { return divider.get_start_time(); }

    //! Set the number of seconds to fold into each sub-integration
    void set_subint_seconds (double subint_seconds)
      { divider.set_seconds (subint_seconds); }

    //! Get the number of seconds to fold into each sub-integration
    double get_subint_seconds () const { return divider.get_seconds(); }

    //! Set the number of turns to fold into each sub-integration
    void set_subint_turns (unsigned subint_turns)
      { divider.set_turns (subint_turns); }

    //! Get the number of turns to fold into each sub-integration
    unsigned get_subint_turns () const { return unsigned(divider.get_turns());}

    void set_fractional_pulses (bool flag) 
      { divider.set_fractional_pulses (flag); }

    /** @name deprecated methods 
     *  Use of these methods is deprecated in favour of attaching
     *  callback methods to the completed event. */
    //@{

    //! Set the file unloader
    void set_unloader (PhaseSeriesUnloader* unloader);

    //! Get the file unloader
    PhaseSeriesUnloader* get_unloader () const { return unloader; }

    //! Decide wether or not to keep the folded profile
    virtual bool keep (PhaseSeries* data) { return true; }

    //@}

    //! Access to the divider
    const TimeDivide* get_divider () const { return &divider; }

    //! Set verbosity ostream
    void set_cerr (std::ostream& os) const;

  protected:

    //! Unload any partially completed integrations
    virtual void unload_partial ();

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

    //! Reset all outputs to null values
    void zero_output ();

    //! Flag set when time divider is initialized
    bool built;

  };

}

#include "dsp/SignalPath.h"

#include "dsp/PhaseSeriesUnloader.h"

#include "Error.h"

//using namespace std;

template <class Op>
dsp::Subint<Op>::Subint ()
{
  built = false;
}

template <class Op>
dsp::Subint<Op>::~Subint ()
{
  if (Op::verbose)
    std::cerr << "dsp::Subint::~Subint" << std::endl;
}

template <class Op>
Op* dsp::Subint<Op>::clone () const
{
  return new Subint<Op>(*this);
}

//! Set verbosity ostream
template <class Op>
void dsp::Subint<Op>::set_cerr (std::ostream& os) const
{
  Operation::set_cerr (os);
  divider.set_cerr (os);
  if (unloader)
    unloader->set_cerr (os);
}

//! Set the file unloader
template <class Op>
void dsp::Subint<Op>::set_unloader (dsp::PhaseSeriesUnloader* _unloader)
{
  if (Op::verbose)
    std::cerr << "dsp::Subint::set_unloader this=" << this 
         << " unloader=" << _unloader << std::endl;

  unloader = _unloader;
  if (unloader)
    unloader->set_cerr (std::cerr);
}

template <class Op>
void dsp::Subint<Op>::prepare ()
{
  if (Op::verbose)
    std::cerr << "dsp::Subint::prepare call Op::prepare" << std::endl;

  Op::prepare ();

  // if unspecified, the first TimeSeries to be folded will define the
  // start time from which to begin cutting up the observation
  if (divider.get_start_time() == MJD::zero)
  {
    if (Op::verbose)
      std::cerr << "dsp::Subint::prepare set divider start time=" 
	   << Op::input->get_start_time() << std::endl;

    if (Op::input->get_start_time() == MJD::zero)
      throw Error (InvalidState, "dsp::Subint::prepare",
		   "input start time not set");

    divider.set_start_time (Op::input->get_start_time());
  }

  if (Op::has_folding_predictor() && divider.get_turns())
    divider.set_predictor (Op::get_folding_predictor());

  built = true;
}


template <class Op>
void dsp::Subint<Op>::transformation () try
{
  if (Op::verbose)
    std::cerr << "dsp::Subint::transformation" << std::endl;

  if (divider.get_turns() == 0 && divider.get_seconds() == 0.0)
    throw Error (InvalidState, "dsp::Subint::tranformation",
		 "sub-integration length not specified");

  if (!built)
    prepare ();

  // flag that the input TimeSeries contains data for another sub-integration
  bool more_data = true;
  bool first_division = true;

  while (more_data)
  {
    divider.set_bounds( Op::get_input() );

    if (!divider.get_fractional_pulses())
      Op::get_output()->set_ndat_expected( divider.get_division_ndat() );

    more_data = divider.get_in_next ();

    if (first_division && divider.get_new_division())
    {
      /* A new division has been started and there is still data in
         the current integration.  This is a sign that the current
         input comes from uncontiguous data, which can arise when
         processing in parallel. */

      unload_partial ();
    }

    if (!divider.get_is_valid())
      continue;

    Op::transformation ();

    if (!divider.get_end_reached())
      continue;

    if (first_division)
    {
      /* When the end of the first division is reached, it is not 100%
         certain that a complete sub-integration is available */

      unload_partial ();
      first_division = false;
      continue;
    }

    if (Op::verbose)
      std::cerr << "dsp::Subint::transformation sub-integration completed" << std::endl;

    PhaseSeries* result = Op::get_result ();

    complete.send (result);

    if (unloader && keep(result))
    {
      if (Op::verbose)
        std::cerr << "dsp::Subint::transformation this=" << this
             << " unloader=" << unloader.get() << std::endl;

      unloader->unload (result);
      zero_output ();
    }
  }
}
catch (Error& error)
{
  throw error += "dsp::Subint::transformation";
}


/*! sets the Fold::idat_start and Fold::ndat_fold attributes */
template <class Op>
void dsp::Subint<Op>::set_limits (const Observation* input)
{
  Op::idat_start = divider.get_idat_start ();
  Op::ndat_fold = divider.get_ndat ();
}

template <class Op>
void dsp::Subint<Op>::finish () try
{
  Op::finish ();

  if (Op::get_result()->get_integration_length() != 0)
  {
    if (Op::verbose)
      std::cerr << "dsp::Subint::finish unload_partial" << std::endl;

    unload_partial ();
  }

  if (unloader)
  {
    if (Op::verbose)
      std::cerr << "dsp::Subint::finish call unloader finish" << std::endl;

    unloader->finish ();
  }
}
catch (Error& error)
{
  throw error += "dsp::Subint::finish";
}

template <class Op>
void dsp::Subint<Op>::unload_partial () try
{
  if (Op::verbose)
    std::cerr << "dsp::Subint::unload_partial to callback" << std::endl;

  PhaseSeries* result = Op::get_result ();

  partial.send (result);

  if (unloader)
  {
    if (Op::verbose)
      std::cerr << "dsp::Subint::unload_partial this=" << this
           << " unloader=" << unloader.get() << std::endl;

    unloader->partial (result);
  }

  zero_output ();
}
catch (Error& error)
{
  throw error += "dsp::Subint::finish";
}

#define SIGNAL_PATH

template <class Op>
void dsp::Subint<Op>::zero_output ()
{
#ifdef SIGNAL_PATH
  SignalPath* path = 0;
  PhaseSeries* out_ptr = Op::output;

  //if (Op::output->has_extensions())
  //  path = Op::output->get_extensions()->get<SignalPath>();
  if (out_ptr->has_extensions())
    path = out_ptr->get_extensions()->get<SignalPath>();

  if (path)
    path->reset();
  else
#endif
    Op::reset();
}

#endif

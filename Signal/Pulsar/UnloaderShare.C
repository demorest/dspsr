/***************************************************************************
 *
 *   Copyright (C) 2007-2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/UnloaderShare.h"
#include "dsp/PhaseSeries.h"
#include "dsp/Operation.h"

#include "ThreadContext.h"
#include "Error.h"

#include <errno.h>
#include <assert.h>

using namespace std;

dsp::UnloaderShare::UnloaderShare (unsigned _contributors)
  : last_division( _contributors, 0 ),
    finished_all( _contributors, false )
{
  context = 0;
  contributors = _contributors;
  wait_all = true;
}

static unsigned max_storage_size = 0;

dsp::UnloaderShare::~UnloaderShare ()
{
  if (Operation::verbose)
    cerr << "dsp::UnloaderShare::~UnloaderShare" << endl;

  cerr << "max_storage_size = " << max_storage_size << endl;
}

void dsp::UnloaderShare::set_context (ThreadContext* c)
{
  context = c;
}

void dsp::UnloaderShare::set_wait_all (bool flag)
{
  wait_all = flag;
}

//! Set the file unloader
void dsp::UnloaderShare::set_unloader (dsp::PhaseSeriesUnloader* _unloader)
{
  unloader = _unloader;
}

//! Get the file unloader
dsp::PhaseSeriesUnloader* dsp::UnloaderShare::get_unloader () const
{
  return unloader;
}

void dsp::UnloaderShare::copy (const TimeDivide* other)
{
  divider = *other;
}

//! Set the start time from which to begin counting sub-integrations
void dsp::UnloaderShare::set_start_time (const MJD& start_time)
{
  divider.set_start_time (start_time);
}

//! Get the start time from which to begin counting sub-integrations
MJD dsp::UnloaderShare::get_start_time () const
{
  return divider.get_start_time(); 
}

//! Get the number of seconds to fold into each sub-integration
double dsp::UnloaderShare::get_subint_seconds () const
{
  return divider.get_seconds();
}

//! Get the number of turns to fold into each sub-integration
unsigned dsp::UnloaderShare::get_subint_turns () const
{
  return unsigned( divider.get_turns() );
}

//! Set the interval over which to fold each sub-integration (in seconds)
void dsp::UnloaderShare::set_subint_seconds (double subint_seconds)
{
  divider.set_seconds (subint_seconds);
}

//! Set the number of pulses to fold into each sub-integration
void dsp::UnloaderShare::set_subint_turns (unsigned subint_turns)
{
  divider.set_turns (subint_turns);
}

void dsp::UnloaderShare::unload (const PhaseSeries* data, Submit* submit)
{
  bool verbose = submit->verbose != 0;

  std::ostream& cerr = *(submit->verbose);
  unsigned contributor = submit->contributor;

  if (verbose)
    cerr << "dsp::UnloaderShare::unload context=" << context << endl;

  ThreadContext::Lock lock (context);

  if (divider.get_turns() == 0 && divider.get_seconds() == 0.0)
    throw Error (InvalidState, "dsp::UnloaderShare::tranformation",
		 "sub-integration length not specified");

  if (!data)
    throw Error (InvalidParam, "dsp::UnloaderShare::tranformation",
		 "PhaseSeries data not provided");

  MJD mid_time = 0.5 * ( data->get_start_time() + data->get_end_time() );
  unsigned division = divider.get_division( mid_time );

  if (verbose)
    cerr << "dsp::UnloaderShare::unload contributor=" << contributor 
	 << " division=" << division << " Nstorage=" << storage.size() << endl;

  last_division[contributor] = division;

  unsigned istore = 0;

  for (istore=0; istore < storage.size(); istore++)
    if (storage[istore]->integrate( contributor, division, data ))
    {
      if (verbose)
        cerr << "dsp::UnloaderShare::unload exit after integrate" << endl;

      context->broadcast ();
      return;
    }

  if (verbose)
    cerr << "dsp::UnloaderShare::unload adding new Storage" << endl;

  Storage* temp = new Storage( contributors, finished_all );
  temp->set_division( division );
  temp->set_finished( contributor );
  temp->set_profiles( data );

  // other contributors may be well ahead of this one
  for (unsigned ic=0; ic < contributors; ic++)
    if ( last_division[ic] > division )
      temp->set_finished( ic );

  storage.push_back( temp );

  if (wait_all)
  {
    temp->wait_all( context );
    unload (temp);
  }
  else
    set_recycle (submit);

  if (verbose)
    cerr << "dsp::UnloaderShare::unload exit" << endl;
}

void dsp::UnloaderShare::set_recycle (Submit* submit)
{
  if (recycle.size() == 0)
  {
    recycle.resize( contributors );
    for (unsigned i=0; i<contributors; i++)
      recycle[i] = new PhaseSeries;
  }

  Reference::To<PhaseSeries> recyclable;

  while (!all_finished())
  {
    submit->to_recycle = get_recyclable();

    if (submit->to_recycle)
      return;

    for (unsigned istore=0; istore < storage.size(); )
    {
      if( storage[istore]->get_finished() )
      {
        set_recyclable (storage[istore]->get_profiles());
	unload (storage[istore]);
      }
      else
	istore ++;
    }  
  }
}

void dsp::UnloaderShare::set_recyclable (const PhaseSeries* data)
{
  for (unsigned i=0; i<contributors; i++)
    if (!recycle[i])
      {
	recycle[i] = data;
	return;
      }

  throw Error (InvalidState, "dsp::UnloaderShare::set_recyclable",
	       "unanticipated state");
}

const dsp::PhaseSeries* dsp::UnloaderShare::get_recyclable ()
{
  for (unsigned i=0; i<contributors; i++)
    if (recycle[i])
      return recycle[i].release();

  return 0;
}

void dsp::UnloaderShare::finish_all (unsigned contributor)
{
  ThreadContext::Lock lock (context);

  if (Operation::verbose)
    cerr << "dsp::UnloaderShare::finish_all"
      " contributor=" << contributor << endl;

  if (finished_all[contributor])
    throw Error (InvalidParam, "dsp::UnloaderShare::finish_all",
                 "contributor %d already finished", contributor);

  finished_all[contributor] = true;

  for (unsigned istore=0; istore < storage.size(); istore++)
    storage[istore]->set_finished (contributor);

  context->signal ();
}

bool dsp::UnloaderShare::all_finished ()
{
  for (unsigned i=0; i<finished_all.size(); i++)
    if (!finished_all[i])
      return false;
  return true;
}

void dsp::UnloaderShare::finish ()
{
  if (Operation::verbose)
    cerr << "dsp::UnloaderShare::finish size=" << storage.size() << endl;

  while( storage.size() )
    unload( storage[0] );

  if (unloader)
    unloader->finish ();
}

void dsp::UnloaderShare::unload (Storage* store)
{
  if (Operation::verbose)
    cerr << "dsp::UnloaderShare::unload store=" << store << endl;

  uint64 division = store->get_division();

  if (unloader) try 
  {
    if (Operation::verbose)
      cerr << "dsp::UnloaderShare::unload unload division=" << division
	   << endl;
    
    unloader->unload( store->get_profiles() );
  }
  catch (Error& error)
    {
      cerr << "dsp::UnloaderShare::unload error unloading division "
	   << division << error;
    }

  for (unsigned i=0; i<storage.size(); i++)
    if (storage[i].get() == store)
    {
      storage.erase( storage.begin() + i );
      break;
    }
}

//! Default constructor
dsp::UnloaderShare::Submit::Submit (UnloaderShare* _parent, unsigned id)
{
  parent = _parent;
  contributor = id;
  verbose = 0;
}

//! Unload the PhaseSeries data
void dsp::UnloaderShare::Submit::unload (const PhaseSeries* profiles)
{
  if (Operation::verbose)
  {
    cerr << "dsp::UnloaderShare::Submit::unload"
      " profiles=" << profiles << " contributor=" << contributor << endl;
    verbose = &cerr;
  }

  parent->unload( profiles, this );
}

//! Return any PhaseSeries to be recycled
dsp::PhaseSeries* dsp::UnloaderShare::Submit::recycle ()
{
  return const_cast<PhaseSeries*>( to_recycle.ptr() );
}

void dsp::UnloaderShare::Submit::finish () try
{
  if (Operation::verbose)
    cerr << "dsp::UnloaderShare::Submit::finish" << endl;

  parent->finish_all( contributor );
}
catch (Error& error)
{
  throw error += "dsp::UnloaderShare::Submit::finish";
}

//! Set the minimum integration length required to unload data
void dsp::UnloaderShare::Submit::set_minimum_integration_length (double secs)
{
  if (!parent->get_unloader())
    throw Error (InvalidState,
		 "dsp::UnloaderShare::Submit::set_minimum_integration_length",
		 "parent unloader not set");

  parent->get_unloader()->set_minimum_integration_length (secs);
}

//! Get submission interface for the specified contributor
dsp::UnloaderShare::Submit* 
dsp::UnloaderShare::new_Submit (unsigned contributor)
{
  return new Submit (this, contributor);
}

//! Default constructor
dsp::UnloaderShare::Storage::Storage (unsigned contributors, 
                                      const std::vector<bool>& all_finished)
  : finished( all_finished )
{
}

dsp::UnloaderShare::Storage::~Storage ()
{
  if (Operation::verbose)
    cerr << "dsp::UnloaderShare::Storage::~Storage" << endl;
}


//! Set the storage area
void dsp::UnloaderShare::Storage::set_profiles( const PhaseSeries* data )
{
  profiles = data;
}

//! Get the storage area
const dsp::PhaseSeries* dsp::UnloaderShare::Storage::get_profiles ()
{
  return profiles;
}

//! Set the division
void dsp::UnloaderShare::Storage::set_division( uint64 d )
{
  division = d;
  if (Operation::verbose)
    print_finished ();
}

//! Get the division
uint64 dsp::UnloaderShare::Storage::get_division ()
{
  return division;
}

//! Add to the contributors that are finished with this integration
bool dsp::UnloaderShare::Storage::integrate( unsigned contributor,
					     uint64 _division,
					     const PhaseSeries* data)
{
  if (contributor >= finished.size())
    throw Error( InvalidParam, "dsp::UnloaderShare::Storage::integrate",
		 "contributor=%d >= size=%d",
		 contributor, finished.size() );

  if (_division == division)
  {
    if (Operation::verbose)
      cerr << "dsp::UnloaderShare::Storage::integrate adding to division="
	   << division << endl;

    /*
      If there is a SignalPath (and assuming that the Fold operation
      is part of the signal path) then the profile data will be
      combined when Fold::combine is called.  Otherwise, the
      PhaseSeries::combine method must be called directly
    */

    PhaseSeries* phase = const_cast<PhaseSeries*>( profiles.get() );

#define SIGNAL_PATH

#ifdef SIGNAL_PATH
    if (profiles->has_extensions())
    {
      SignalPath* into = phase->get_extensions()->get<SignalPath>();
      const SignalPath* from = data->get_extensions()->get<SignalPath>();

      if (into && from)
      {
	if (Operation::verbose)
	  cerr << "dsp::UnloaderShare::Storage::integrate into "
	    "profile=" << profiles.get() << " list=" << into->get_list()
	       << "\ndsp::UnloaderShare::Storage::integrate from "
	    "profile=" << data << " list=" << from->get_list() << endl;

	into->combine(from);
      }
    }
    else
#endif
      phase->combine( data );

    set_finished( contributor );

    return true;
  }

  if (_division > division)
    set_finished( contributor );

  return false;
}

void dsp::UnloaderShare::Storage::wait_all (ThreadContext* context)
{
  while (!get_finished())
    context->wait();
}

void dsp::UnloaderShare::Storage::set_finished (unsigned contributor)
{
  finished[contributor] = true;
}

//! Return true when all contributors are finished with this integration
bool dsp::UnloaderShare::Storage::get_finished ()
{
  if (Operation::verbose)
    print_finished ();

  for (unsigned i=0; i < finished.size(); i++)
    if (!finished[i])
      return false;

  return true;
}

void dsp::UnloaderShare::Storage::print_finished ()
{
  cerr << "division=" << division << " finished:";
  for (unsigned i=0; i < finished.size(); i++)
    cerr << " " << finished[i];
  cerr << endl;
}



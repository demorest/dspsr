/***************************************************************************
 *
 *   Copyright (C) 2007-2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/UnloaderShare.h"
#include "dsp/PhaseSeries.h"
#include "dsp/PhaseSeriesUnloader.h"
#include "dsp/Operation.h"

#include "ThreadContext.h"
#include "Error.h"

using namespace std;

dsp::UnloaderShare::UnloaderShare (unsigned _contributors)
  : last_division( _contributors, 0 ),
    finished_all( _contributors, false )
{
  context = 0;
  contributors = _contributors;
  wait_all = true;
}

dsp::UnloaderShare::~UnloaderShare ()
{
}

void dsp::UnloaderShare::set_context (ThreadContext* c)
{
  context = c;
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

void dsp::UnloaderShare::unload (const PhaseSeries* data,
				 unsigned contributor)
{
  if (Operation::verbose)
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

  if (Operation::verbose)
    cerr << "dsp::UnloaderShare::unload contributor=" << contributor 
	 << " division=" << division << " Nstorage=" << storage.size() << endl;

  last_division[contributor] = division;

  unsigned istore = 0;
  for (istore=0; istore < storage.size(); istore++)
    if (storage[istore]->integrate( contributor, division, data ))
      break;

  if (istore == storage.size())
  {
    if (Operation::verbose)
      cerr << "dsp::UnloaderShare::unload adding new Storage" << endl;

    Storage* temp = new Storage( contributors, finished_all );
    temp->set_division( division );
    temp->set_finished( contributor );

    // other contributors may be well ahead of this one
    for (unsigned ic=0; ic < contributors; ic++)
      if ( last_division[ic] > division )
        temp->set_finished( ic );

    if (wait_all)
      temp->set_profiles( const_cast<PhaseSeries*>(data) );
    else
      temp->set_profiles( data->clone() );

    storage.push_back( temp );

    if (wait_all)
    {
      temp->wait_all( context );
      unload (temp);
    }
  }

  if (wait_all)
    return;

  istore=0;
  while( istore < storage.size() )
  {
    if( storage[istore]->get_finished() )
      unload (storage[istore]);
    else
      istore ++;
  }  

  if (Operation::verbose)
    cerr << "dsp::UnloaderShare::unload exit" << endl;
}

void dsp::UnloaderShare::finish_all (unsigned contributor)
{
  ThreadContext::Lock lock (context);

  if (Operation::verbose)
    cerr << "dsp::UnloaderShare::finish_all contributor=" << contributor << endl;

  if (finished_all[contributor])
    throw Error (InvalidParam, "dsp::UnloaderShare::finish_all",
                 "contributor %d already finished", contributor);

  finished_all[contributor] = true;

  for (unsigned istore=0; istore < storage.size(); istore++)
    storage[istore]->set_finished (contributor);
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
  list = 0;
}

//! Set the signal path that yielded the folded PhaseSeries data
void dsp::UnloaderShare::Submit::set_list (SignalPath::List* _list)
{
  list = _list;
}

//! Unload the PhaseSeries data
void dsp::UnloaderShare::Submit::unload (const PhaseSeries* profiles)
{
  if (Operation::verbose)
    cerr << "dsp::UnloaderShare::Submit::unload"
      " profiles=" << profiles << " contributor=" << contributor << endl;

#ifdef SIGNAL_PATH
  if (list)
  {
    SignalPath* p = const_cast<PhaseSeries*>(profiles)->getadd<SignalPath>();
    p->set_list (list);

    if (Operation::verbose)
      cerr << "dsp::UnloaderShare::Submit::unload"
              " this=" << this << " list=" << list << endl;
  }
#endif

  parent->unload( profiles, contributor );
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
  context = 0;
}

dsp::UnloaderShare::Storage::~Storage ()
{
  if (Operation::verbose)
    cerr << "dsp::UnloaderShare::Storage::~Storage" << endl;
}


//! Set the storage area
void dsp::UnloaderShare::Storage::set_profiles( PhaseSeries* data )
{
  profiles = data;
}

//! Get the storage area
dsp::PhaseSeries* dsp::UnloaderShare::Storage::get_profiles ()
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

#ifdef SIGNAL_PATH
    if (profiles->has<SignalPath>() && data->has<SignalPath>())
    {
      SignalPath* into = profiles->get<SignalPath>();
      const SignalPath* from = data->get<SignalPath>();

      if (Operation::verbose)
        cerr << "dsp::UnloaderShare::Storage::integrate into "
          "profile=" << profiles.get() << " list=" << into->get_list() << endl
             << "dsp::UnloaderShare::Storage::integrate from "
          "profile=" << data << " list=" << from->get_list() << endl;

      into->combine(from);
    }
    else
#endif
      profiles->combine( data );

    set_finished( contributor );

    return true;
  }

  if (_division > division)
    set_finished( contributor );

  return false;
}

void dsp::UnloaderShare::Storage::wait_all (ThreadContext* ctxt)
{
  context = ctxt;
  while (!get_finished())
    context->wait();
  context = 0;
}

void dsp::UnloaderShare::Storage::set_finished (unsigned contributor)
{
  finished[contributor] = true;
  if (context)
    context->signal();
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



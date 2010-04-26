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

dsp::UnloaderShare::~UnloaderShare ()
{
  if (Operation::verbose)
    cerr << "dsp::UnloaderShare::~UnloaderShare" << endl;
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

static unsigned max_storage_size = 0;

void dsp::UnloaderShare::unload (const PhaseSeries* data, Submit* submit) try
{
  std::ostream* verbose = 0;
  if (Operation::verbose)
    verbose = &(submit->cerr);

  unsigned contributor = submit->contributor;

  std::ostream& cerr = *verbose;

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
      break;

  if (istore < storage.size())
  {
    // wake up any threads waiting for completion
    if (wait_all)
      for (istore=0; istore < storage.size(); istore++)
        if (storage[istore]->get_finished ())
        {
          context->broadcast ();
          break;
        }
  }
  else if (data->get_integration_length() > 0)
  {
    if (verbose)
      cerr << "dsp::UnloaderShare::unload adding new Storage" << endl;

    Storage* temp = new Storage( contributors, finished_all );
    temp->set_division( division );
    temp->set_finished( contributor );

    // other contributors may be well ahead of this one
    for (unsigned ic=0; ic < contributors; ic++)
      if ( last_division[ic] > division )
        temp->set_finished( ic );

    if (!wait_all)
      temp->set_profiles( data->clone() );
    else
      temp->set_profiles( data );

    storage.push_back( temp );

    if (wait_all)
    {
      temp->wait_all( context );
      unload (temp);
    }

  }

  if (wait_all)
    return;

  if (storage.size() > max_storage_size)
    max_storage_size = storage.size();

  istore=0;
  while( istore < storage.size() )
  {
    if( storage[istore]->get_finished() )
      //unload (storage[istore]); 
      nonblocking_unload (istore, submit);
    else
      istore ++;
  }  

  if (verbose)
    cerr << "dsp::UnloaderShare::unload exit" << endl;
}
catch (Error& error)
{
  throw error += "dsp::UnloaderShare::unload";
}

void dsp::UnloaderShare::finish_all (unsigned contributor)
{
  if (Operation::verbose)
    cerr << "dsp::UnloaderShare::finish_all contributor=" << contributor << endl;

  ThreadContext::Lock lock (context);

  if (finished_all[contributor])
    throw Error (InvalidParam, "dsp::UnloaderShare::finish_all",
                 "contributor %d already finished", contributor);

  finished_all[contributor] = true;

  for (unsigned istore=0; istore < storage.size(); istore++)
    storage[istore]->set_finished (contributor);

  if (wait_all)
    context->broadcast ();
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
  ThreadContext::Lock lock (context);

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

  uint64_t division = store->get_division();

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

//! Unload the storage in parallel
void dsp::UnloaderShare::nonblocking_unload (unsigned istore, Submit* submit)
{
  if (Operation::verbose)
    submit->cerr << "dsp::UnloaderShare::nonblocking_unload" << endl;

  Reference::To<Storage> store = storage[istore];
  storage.erase (storage.begin() + istore);

  context->unlock ();

  uint64_t division = store->get_division();

  try {

    if (Operation::verbose)
      submit->cerr << "dsp::UnloaderShare::nonblocking_unload division="
                   << division << endl;
    
    submit->unloader->unload( store->get_profiles() );
  }
  catch (Error& error)
  {
    submit->cerr << "dsp::UnloaderShare::nonblocking_unload error division "
         << division << error;
  }

  context->lock ();
}

//! Default constructor
dsp::UnloaderShare::Submit::Submit (UnloaderShare* _parent, unsigned id)
{
  parent = _parent;
  contributor = id;
}

dsp::UnloaderShare::Submit* dsp::UnloaderShare::Submit::clone () const
{
  return 0;
}

//! Set verbosity ostream
void dsp::UnloaderShare::Submit::set_cerr (std::ostream& os) const
{
  PhaseSeriesUnloader::set_cerr (os);
  if (unloader)
    unloader->set_cerr (os);
}

//! Set the file unloader
void dsp::UnloaderShare::Submit::set_unloader (dsp::PhaseSeriesUnloader* psu)
{
  unloader = psu;
  if (unloader)
    unloader->set_cerr (cerr);
}

//! Unload the PhaseSeries data
void dsp::UnloaderShare::Submit::unload (const PhaseSeries* profiles) try
{
  if (Operation::verbose)
    cerr << "dsp::UnloaderShare::Submit::unload"
      " profiles=" << profiles << " contributor=" << contributor << endl;

  if (unloader)
    unloader->unload (profiles);
  else
    parent->unload( profiles, this );
}
catch (Error& error)
{
  throw error += "dsp::UnloaderShare::Submit::unload";
}

void dsp::UnloaderShare::Submit::partial (const PhaseSeries* profiles) try
{
  if (Operation::verbose)
    cerr << "dsp::UnloaderShare::Submit::partial"
      " profiles=" << profiles << " contributor=" << contributor << endl;

  parent->unload( profiles, this );
}
catch (Error& error)
{
  throw error += "dsp::UnloaderShare::Submit::partial";
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
    std::cerr << "dsp::UnloaderShare::Storage::~Storage" << endl;
}


//! Set the storage area
void dsp::UnloaderShare::Storage::set_profiles (const PhaseSeries* data)
{
  profiles = const_cast<PhaseSeries*>( data );
}

//! Get the storage area
dsp::PhaseSeries* dsp::UnloaderShare::Storage::get_profiles ()
{
  return profiles;
}

//! Set the division
void dsp::UnloaderShare::Storage::set_division( uint64_t d )
{
  division = d;
  if (Operation::verbose)
    print_finished ();
}

//! Get the division
uint64_t dsp::UnloaderShare::Storage::get_division ()
{
  return division;
}

//! Add to the contributors that are finished with this integration
bool dsp::UnloaderShare::Storage::integrate( unsigned contributor,
					     uint64_t _division,
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

#define SIGNAL_PATH

#ifdef SIGNAL_PATH
    if (profiles->has_extensions())
    {
      SignalPath* into = profiles->get_extensions()->get<SignalPath>();
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
      profiles->combine (data);

    set_finished( contributor );

    return true;
  }

  if (_division > division)
    set_finished( contributor );

  return false;
}

void dsp::UnloaderShare::Storage::wait_all (ThreadContext* context)
{
  if (Operation::verbose)
    cerr << "dsp::UnloaderShare::Storage::wait_all" << endl;
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



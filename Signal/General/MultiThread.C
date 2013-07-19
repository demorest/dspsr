/***************************************************************************
 *
 *   Copyright (C) 2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/MultiThread.h"

#include "dsp/Input.h"
#include "dsp/InputBufferingShare.h"

#include "FTransformAgent.h"
#include "ThreadContext.h"

#include <fstream>
#include <stdlib.h>
#include <errno.h>

using namespace std;

//! Constructor
dsp::MultiThread::MultiThread ()
{
  input_context = new ThreadContext;
  state_changes = new ThreadContext;

  if (!FTransform::Agent::context)
    FTransform::Agent::context = new ThreadContext;
}
    
//! Destructor
dsp::MultiThread::~MultiThread ()
{
  delete input_context;
  delete state_changes;
}

//! Set the number of thread to be used
void dsp::MultiThread::set_nthread (unsigned nthread)
{
  threads.resize (nthread);
  for (unsigned i=0; i<nthread; i++)
    if (!threads[i])
      threads[i] = new_thread();

  if (configuration)
    set_configuration (configuration);

  if (input)
    set_input (input);
}


//! Set the configuration to be used in prepare and run
void dsp::MultiThread::set_configuration (SingleThread::Config* config)
{
  configuration = config;

  for (unsigned i=0; i<threads.size(); i++)
    threads[i]->set_configuration( config );
}

//! Set the Input from which data will be read
void dsp::MultiThread::set_input (Input* _input)
{
  if (Operation::verbose)
    cerr << "dsp::MultiThread::set_input input=" << _input << endl;

  input = _input;

  if (!input)
    return;

  input->set_context( input_context );

  for (unsigned i=0; i<threads.size(); i++)
  {
    threads[i]->set_input( input );
    threads[i]->input_context = input_context;
  }
}

dsp::Input* dsp::MultiThread::get_input ()
{
  return input;
}


void dsp::MultiThread::construct ()
{
  launch_threads ();

  for (unsigned i=0; i<threads.size(); i++)
  {
    threads[i]->thread_id = i;
    dsp::MultiThread::signal (threads[i], SingleThread::Construct);
    dsp::MultiThread::wait (threads[i], SingleThread::Constructed);
  }

  share ();

  for (unsigned i=1; i<threads.size(); i++)
    threads[i]->colleague = threads[0];

}

void dsp::MultiThread::prepare ()
{
  // thread 0 will prepare as normal
  // the rest will call SingleThread::share and then prepare

  for (unsigned i=0; i<threads.size(); i++)
  {
    dsp::MultiThread::signal (threads[i], SingleThread::Prepare);
    dsp::MultiThread::wait (threads[i], SingleThread::Prepared);
  }
}

void dsp::MultiThread::share ()
{
  if (Operation::verbose)
    cerr << "dsp::MultiThread::share installing InputBuffering::Share" << endl;

  //
  // install InputBuffering::Share policy
  //
  typedef Transformation<TimeSeries,TimeSeries> Xform;

  for (unsigned iop=0; iop < threads[0]->operations.size(); iop++)
  {
    Xform* xform = dynamic_kast<Xform>( threads[0]->operations[iop] );

    if (!xform)
      continue;

    if (!xform->has_buffering_policy())
      continue;

    InputBuffering* ibuf;
    ibuf = dynamic_cast<InputBuffering*>( xform->get_buffering_policy() );

    if (!ibuf)
      continue;

    xform->set_buffering_policy( new InputBuffering::Share (ibuf, xform) );
  }
}

uint64_t dsp::MultiThread::get_minimum_samples () const
{
  if (threads.size() == 0)
    return 0;
  return threads[0]->get_minimum_samples();
}

void dsp::MultiThread::wait (SingleThread* thread, SingleThread::State state)
{
  ThreadContext::Lock lock (thread->state_change);
  while ( thread->state != state )
    thread->state_change->wait ();
}

void dsp::MultiThread::signal (SingleThread* thread, SingleThread::State state)
{
  ThreadContext::Lock lock (thread->state_change);
  thread->state = state;
  thread->state_change->broadcast();
}

void* dsp::MultiThread::thread (void* context) try
{
  dsp::SingleThread* thread = reinterpret_cast<dsp::SingleThread*>( context );

  // Construct

  wait (thread, SingleThread::Construct);
  thread->construct ();
  signal (thread, SingleThread::Constructed);

  // Prepare

  wait (thread, SingleThread::Prepare);
  if (thread->colleague)
    thread->share (thread->colleague);
  thread->prepare ();
  signal (thread, SingleThread::Prepared);

  // Run

  wait (thread, SingleThread::Run);

  SingleThread::State state = SingleThread::Done;

  try
  {
    if (thread->log) *(thread->log) << "THREAD STARTED" << endl;

    thread->run();

    if (thread->log) *(thread->log) << "THREAD run ENDED" << endl;
  }
  catch (Error& error)
  {
    if (thread->log) *(thread->log) << "THREAD ERROR: " << error << endl;

    cerr << "THREAD ERROR: " << error << endl;

    state = SingleThread::Fail;
    thread->error = error;

    exit (-1);
  }

  if (thread->log) *(thread->log) << "SIGNAL end state" << endl;

  signal (thread, state);

  if (thread->log) *(thread->log) << "THREAD EXIT" << endl;

  pthread_exit (0);
}
catch (Error& error)
{
  cerr << "THREAD ERROR: " << error << endl;
  exit (-1);
}

//! Run through the data
void dsp::MultiThread::run ()
{
  ThreadContext::Lock lock (state_changes);

  for (unsigned i=0; i<threads.size(); i++)
    threads[i]->state = SingleThread::Run;

  state_changes->broadcast();
}

void dsp::MultiThread::launch_threads ()
{
  ids.resize( threads.size() );

  for (unsigned i=0; i<threads.size(); i++)
  {
    SingleThread* a_thread = threads[i];

    if (Operation::verbose)
    {
      string logname = "dspsr.log." + tostring (i);

      cerr << "dsp::MultiThread::run spawning thread " << i 
	   << " ptr=" << a_thread << " log=" << logname << endl;

      a_thread->take_ostream( new std::ofstream (logname.c_str()) );
    }

    a_thread->state = SingleThread::Idle;
    a_thread->state_change = state_changes;

    errno = pthread_create (&ids[i], 0, thread, a_thread);

    if (errno != 0)
      throw Error (FailedSys, "psr::MultiThread::run", "pthread_create");

  }

  if (Operation::verbose)
    cerr << "psr::MultiThread::run all threads spawned" << endl;

}

//! Finish everything
void dsp::MultiThread::finish ()
{
  Error error (InvalidState, "");

  unsigned errors = 0;
  unsigned finished = 0;

  SingleThread* first = 0;

  ThreadContext::Lock lock (state_changes);

  while (finished < threads.size())
  {
    for (unsigned i=0; i<threads.size(); i++)
    {
      SingleThread::State state = threads[i]->state;

      if (state != SingleThread::Done && state != SingleThread::Fail)
      {
        if (Operation::verbose)
        {
          cerr << "dsp::MultiThread::finish thread " << i;
          if (state == SingleThread::Run)
            cerr << " pending" << endl;
          else if (state == SingleThread::Joined)
            cerr << " processed" << endl;
	  else
	    cerr << " unknown state" << endl;
        }
        continue;
      }

      try {

	if (Operation::verbose)
	  cerr << "psr::MultiThread::finish joining thread " << i << endl;

	void* result = 0;
	pthread_join (ids[i], &result);

        if (Operation::verbose)
          cerr << "psr::MultiThread::finish thread " << i << " joined" << endl;

	finished ++;
        threads[i]->state = SingleThread::Joined;

	if (state == SingleThread::Fail)
        {
	  errors ++;
	  error = threads[i]->error;
          cerr << "thread " << i << " aborted with error\n\t"
               << error.get_message() << endl;
          continue;
	}

	if (!first)
	{
          if (Operation::verbose)
            cerr << "psr::MultiThread::finish initializing first" << endl;

	  first = threads[i];
	}
	else
        {
          if (Operation::verbose)
            cerr << "psr::MultiThread::finish combining with first" << endl;

	  first->combine( threads[i] );
        }

      }
      catch (Error& error)
      {
        cerr << "psr::MultiThread::finish failure on thread " 
             << i << error << endl;
      }

    }

    if (finished < threads.size())
    {
      if (Operation::verbose)
        cerr << "psr::MultiThread::finish wait on condition" << endl;

      state_changes->wait();

      if (Operation::verbose)
        cerr << "psr::MultiThread::finish condition wait returned" << endl;
    }

  }

  if (first)
  {
    if (Operation::verbose)
      cerr << "psr::MultiThread::finish via first" << endl;

    first->finish();
  }

  if (errors)
  {
    error << errors << " threads aborted with an error";
    throw error += "dsp::MultiThread::finish";
  }
}



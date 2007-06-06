/***************************************************************************
 *
 *   Copyright (C) 2007 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/LoadToFoldN.h"
#include "dsp/LoadToFoldConfig.h"

#include "dsp/Input.h"
#include "dsp/InputBufferingShare.h"

#include "dsp/Scratch.h"
#include "dsp/Dedispersion.h"
#include "dsp/Fold.h"

#include "FTransformAgent.h"
#include "ThreadContext.h"

#include <fstream>
#include <errno.h>

using namespace std;

//! Constructor
dsp::LoadToFoldN::LoadToFoldN (unsigned nthread)
{
  input_context = new ThreadContext;
  completion = new ThreadContext;

  if (nthread)
    set_nthread (nthread);

  if (!FTransform::Agent::context)
    FTransform::Agent::context = new ThreadContext;
}
    
//! Destructor
dsp::LoadToFoldN::~LoadToFoldN ()
{
  delete input_context;
}

//! Set the number of thread to be used
void dsp::LoadToFoldN::set_nthread (unsigned nthread)
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
void dsp::LoadToFoldN::set_configuration (Config* config)
{
  configuration = config;
  for (unsigned i=0; i<threads.size(); i++)
    threads[i]->set_configuration( config );
}

//! Set the Input from which data will be read
void dsp::LoadToFoldN::set_input (Input* _input)
{
  input = _input;

  if (!input)
    return;

  input->set_context( input_context );

  for (unsigned i=0; i<threads.size(); i++)
    threads[i]->set_input( input );
}

//! Prepare to fold the input TimeSeries
void dsp::LoadToFoldN::prepare ()
{
  if (! threads.size() )
    throw Error (InvalidState, "dsp::LoadToFoldN::prepare", "no threads");

  threads[0]->prepare ();

  if (threads[0]->kernel && !threads[0]->kernel->context)
    threads[0]->kernel->context = new ThreadContext;

  threads[0]->report = threads.size();

  //
  // install InputBuffering::Share policy
  //
  typedef Transformation<TimeSeries,TimeSeries> Xform;

  for (unsigned iop=0; iop < threads[0]->operations.size(); iop++) {

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

  for (unsigned i=1; i<threads.size(); i++) {

    //
    // clone the Fold/SubFold operations (share Pulsar::Predictor)
    //
    // This code satisfies two preconditions:
    // 1) the folding operation may be either a Fold or SubFold class
    // 2) the folding operations should share predictors but not outputs
    //

    unsigned nfold = threads[0]->fold.size();
    threads[i]->fold.resize( nfold );
    for (unsigned ifold = 0; ifold < nfold; ifold ++)  {
      // the clone automatically copies the pointers to predictors ...
      threads[i]->fold[ifold] = threads[0]->fold[ifold]->clone();
      // ... and the outputs.  New ones will be created in prepare()
      threads[i]->fold[ifold]->set_output( 0 );
    }

    //
    // share the dedispersion kernel
    //
    threads[i]->kernel = threads[0]->kernel;

    //
    // only the first thread prints updates
    //
    threads[i]->report = 0;

    if (Operation::verbose)
      cerr << "dsp::LoadToFoldN::prepare preparing thread " << i << endl;

    threads[i]->prepare ();

    //
    // share buffering policies
    //
    typedef Transformation<TimeSeries,TimeSeries> Xform;

    for (unsigned iop=0; iop < threads[i]->operations.size(); iop++) {

      Xform* trans0 = dynamic_kast<Xform>( threads[0]->operations[iop] );

      if (!trans0)
	continue;

      if (!trans0->has_buffering_policy())
	continue;

      InputBuffering::Share* ibuf0;
      ibuf0 = dynamic_cast<InputBuffering::Share*>
	( trans0->get_buffering_policy() );

      if (!ibuf0)
	continue;

      Xform* trans = dynamic_kast<Xform>( threads[i]->operations[iop] );

      if (!trans)
	throw Error (InvalidState, "dsp::LoadToFoldN::prepare",
		     "mismatched operation type");

      if (!trans->has_buffering_policy())
	throw Error (InvalidState, "dsp::LoadToFoldN::prepare",
		     "mismatched buffering policy");

      // cerr << "Sharing buffering policy of " << trans->get_name() << endl;

      trans->set_buffering_policy( ibuf0->clone(trans) );

    }

  }

}

uint64 dsp::LoadToFoldN::get_minimum_samples () const
{
  if (threads.size() == 0)
    return 0;
  return threads[0]->get_minimum_samples();
}

void* dsp::LoadToFoldN::thread (void* context)
{
  dsp::LoadToFold1* fold = reinterpret_cast<dsp::LoadToFold1*>( context );

  try {

    if (fold->log) *(fold->log) << "THREAD STARTED" << endl;
  
    fold->run();

    fold->status = 2;

    if (fold->log) *(fold->log) << "THREAD ENDED" << endl;

  }
  catch (Error& error) {

    if (fold->log) *(fold->log) << "THREAD ERROR: " << error << endl;

    fold->status = -1;
    fold->error = error;

  }

  //
  // the lock must go out of scope before the signal will be delivered
  {
    ThreadContext::Lock lock (fold->completion);
    fold->completion->signal();
  }

  pthread_exit (0);
}

//! Run through the data
void dsp::LoadToFoldN::run ()
{
  ids.resize( threads.size() );

  for (unsigned i=0; i<threads.size(); i++) {

    LoadToFold1* fold = threads[i];

    if (Operation::verbose) {

      string logname = "dspsr.log." + tostring (i);

      cerr << "dsp::LoadToFoldN::run spawning thread " << i 
	   << " ptr=" << fold << " log=" << logname << endl;

      fold->take_ostream( new std::ofstream (logname.c_str()) );

    }

    fold->status = 1;
    fold->completion = completion;

    errno = pthread_create (&ids[i], 0, thread, fold);

    if (errno != 0)
      throw Error (FailedSys, "psr::LoadToFoldN::run", "pthread_create");

  }

  if (Operation::verbose)
    cerr << "psr::LoadToFoldN::run all threads spawned" << endl;

}

//! Finish everything
void dsp::LoadToFoldN::finish ()
{
  Error error (InvalidState, "");

  unsigned errors = 0;
  unsigned finished = 0;

  unsigned first = 0;

  ThreadContext::Lock lock (completion);

  while (finished < threads.size()) {

    for (unsigned i=0; i<threads.size(); i++) {

      if (threads[i]->status == 2 || threads[i]->status == -1) {

	if (Operation::verbose)
	  cerr << "psr::LoadToFoldN::finish joining thread " << i << endl;

	void* result = 0;
	pthread_join (ids[i], &result);

	if (threads[i]->status < 0) {
	  errors ++;
	  error = threads[i]->error;
	}

	threads[i]->status = 0;

	if (finished == 0)
	  first = i;

	else if (!configuration->single_pulse)
	  for (unsigned ifold=0; ifold<threads[i]->fold.size(); ifold++)
	    *( threads[first]->fold[ifold]->get_output() ) +=
	      *( threads[first]->fold[ifold]->get_output() );

	finished ++;

      }
      else if (Operation::verbose) {

	cerr << "dsp::LoadToFoldN::finish thread " << i;
	if (threads[i]->status == 1)
	  cerr << " pending" << endl;
	else
	  cerr << " processed" << endl;

      }

    }

    if (finished < threads.size())
      completion->wait();

  }

  threads[first]->finish();

  if (errors) {
    error << errors << " threads aborted with an error";
    throw error += "dsp::LoadToFoldN::finish";
  }

  // add folded profiles together XXX
}

//! The creator of new LoadToFold1 threadss
dsp::LoadToFold1* dsp::LoadToFoldN::new_thread ()
{
  return new LoadToFold1;
}

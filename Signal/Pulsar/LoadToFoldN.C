/***************************************************************************
 *
 *   Copyright (C) 2007 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/LoadToFoldN.h"
#include "dsp/LoadToFoldConfig.h"

#include "dsp/Input.h"
#include "dsp/Scratch.h"
#include "dsp/Dedispersion.h"
#include "dsp/Fold.h"

#include "FTransform.h"
#include "ThreadContext.h"

#include <fstream>
using namespace std;

#include <errno.h>

//! Constructor
dsp::LoadToFoldN::LoadToFoldN (unsigned nthread)
{
  input_context = new ThreadContext;

  if (nthread)
    set_nthread (nthread);

  if (!FTransform::context)
    FTransform::context = new ThreadContext;

  if (Operation::verbose)
    cerr << "FTransform::context set to " << FTransform::context << endl;
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

    if (fold->log) *(fold->log) << "THREAD ENDED" << endl;

  }
  catch (Error& error) {

    if (fold->log) *(fold->log) << "THREAD ERROR: " << error << endl;

    pthread_exit (0);

  }

  pthread_exit (context);
}

//! Run through the data
void dsp::LoadToFoldN::run ()
{
  ids.resize( threads.size() );

  for (unsigned i=0; i<threads.size(); i++) {

    LoadToFold1* ptr = threads[i].ptr();

    if (Operation::verbose) {

      string logname = "dspsr.log." + tostring (i);

      cerr << "dsp::LoadToFoldN::run spawning thread " << i 
	   << " ptr=" << ptr << " log=" << logname << endl;

      ptr->take_ostream( new std::ofstream (logname.c_str()) );

    }

    errno = pthread_create (&ids[i], 0, thread, ptr);

    if (errno != 0)
      throw Error (FailedSys, "psr::LoadToFoldN::run", "pthread_create");

  }

  if (Operation::verbose)
    cerr << "psr::LoadToFoldN::run all threads spawned" << endl;

}

//! Finish everything
void dsp::LoadToFoldN::finish ()
{
  unsigned errors = 0;

  for (unsigned i=0; i<threads.size(); i++) {

    if (Operation::verbose)
      cerr << "psr::LoadToFoldN::finish joining thread " << i << endl;

    void* result = 0;
    pthread_join (ids[i], &result);

    if (result != threads[i].ptr())
      errors ++;

  }

  if (errors)
    throw Error (InvalidState, "dsp::LoadToFoldN::finish",
		 "%d threads aborted with an error", errors);

  // add folded profiles together XXX
}

//! The creator of new LoadToFold1 threadss
dsp::LoadToFold1* dsp::LoadToFoldN::new_thread ()
{
  return new LoadToFold1;
}

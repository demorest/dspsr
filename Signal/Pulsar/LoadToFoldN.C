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

#include "ThreadContext.h"

//! Constructor
dsp::LoadToFoldN::LoadToFoldN ()
{
  context = 0;
}
    
//! Destructor
dsp::LoadToFoldN::~LoadToFoldN ()
{
  delete context;
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

  if (!context)
    context = new ThreadContext;

  input->set_context( context );

  for (unsigned i=0; i<threads.size(); i++)
    threads[i]->set_input( input );
}

//! Prepare to fold the input TimeSeries
void dsp::LoadToFoldN::prepare ()
{
  if (! threads.size() )
    throw Error (InvalidState, "dsp::LoadToFoldN::prepare", "no threads");

  threads[0]->prepare ();

  for (unsigned i=1; i<threads.size(); i++) {

  // loop and copy XXX

  }


}

static void* thread_wrapper (void* context)
{
  reinterpret_cast<dsp::LoadToFold1*>( context )->run();
}

//! Run through the data
void dsp::LoadToFoldN::run ()
{

}

//! Finish everything
void dsp::LoadToFoldN::finish ()
{
  // wait until threads are finished
  // add folded profiles together
}

//! The creator of new LoadToFold1 threads
dsp::LoadToFold1* dsp::LoadToFoldN::new_thread ()
{
  return new LoadToFold1;
}

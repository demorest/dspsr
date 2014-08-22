/***************************************************************************
 *
 *   Copyright (C) 2014 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/LoadToFilN.h"

#include "dsp/Input.h"
#include "dsp/InputBufferingShare.h"

#include "dsp/Dedispersion.h"
#include "dsp/OutputFile.h"
#include "dsp/OutputFileShare.h"
#include "FTransformAgent.h"
#include "ThreadContext.h"

#include <fstream>
#include <stdlib.h>
#include <errno.h>

using namespace std;

//! Constructor
dsp::LoadToFilN::LoadToFilN (LoadToFil::Config* config)
{
  configuration = config;
  set_nthread (configuration->get_total_nthread());
}
    
//! Set the number of thread to be used
void dsp::LoadToFilN::set_nthread (unsigned nthread)
{

  if (configuration) 
  {
    if (nthread>1 && configuration->rescale_seconds>0.0) 
    {
      cerr << "dsp::LoadToFilN::set_nthread WARNING disabling rescaling"
        << endl;
      configuration->rescale_seconds = 0.0;
    }
  }

  MultiThread::set_nthread (nthread);

  FTransform::nthread = nthread;

  if (configuration)
    set_configuration (configuration);
}

dsp::LoadToFil* dsp::LoadToFilN::at (unsigned i)
{
  return dynamic_cast<LoadToFil*>( threads.at(i).get() );
}

//! Set the configuration to be used in prepare and run
void dsp::LoadToFilN::set_configuration (LoadToFil::Config* config)
{
  configuration = config;

  MultiThread::set_configuration (config);

  for (unsigned i=0; i<threads.size(); i++)
    at(i)->set_configuration( config );
}

void dsp::LoadToFilN::share ()
{
  MultiThread::share ();

  if (at(0)->kernel && !at(0)->kernel->context)
    at(0)->kernel->context = new ThreadContext;

  // Output file sharing
  output_file = new OutputFileShare(threads.size());
  output_file->set_context(new ThreadContext);
  output_file->set_output_file(at(0)->outputFile);

  // Replace the normal output with shared version in each thread
  for (unsigned i=0; i<threads.size(); i++) 
  {
    at(i)->operations.pop_back(); // unload should be last....
    OutputFileShare::Submit* sub = output_file->new_Submit(i);
    sub->set_input(at(i)->outputFile->get_input());
    at(i)->operations.push_back(sub);
  }

}

void dsp::LoadToFilN::finish ()
{
  MultiThread::finish ();

  //for (unsigned i=0; i<unloader.size(); i++)
  //{
  //  if (Operation::verbose)
  //    cerr << "psr::LoadToFoldN::finish unloader[" << i << "]" << endl;
  //
  //  unloader[i]->finish();
  //}
}

//! The creator of new LoadToFil threadss
dsp::LoadToFil* dsp::LoadToFilN::new_thread ()
{
  return new LoadToFil;
}


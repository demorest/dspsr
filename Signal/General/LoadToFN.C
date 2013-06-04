/***************************************************************************
 *
 *   Copyright (C) 2007-2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/LoadToF1.h"
#include "dsp/LoadToFN.h"

#include "dsp/Input.h"
#include "dsp/InputBufferingShare.h"

#include "FTransformAgent.h"
//#include "ThreadContext.h"

#include <fstream>
#include <stdlib.h>
#include <errno.h>

using namespace std;

//! Constructor
dsp::LoadToFN::LoadToFN (LoadToF::Config* config)
{
  configuration = config;
  set_nthread (configuration->get_total_nthread());
}
    
//! Set the number of thread to be used
void dsp::LoadToFN::set_nthread (unsigned nthread)
{
  MultiThread::set_nthread (nthread);

  FTransform::nthread = nthread;

  if (configuration)
    set_configuration (configuration);
}

dsp::LoadToF* dsp::LoadToFN::at (unsigned i)
{
  return dynamic_cast<LoadToF*>( threads.at(i).get() );
}

//! Set the configuration to be used in prepare and run
void dsp::LoadToFN::set_configuration (LoadToF::Config* config)
{
  configuration = config;

  MultiThread::set_configuration (config);

  for (unsigned i=0; i<threads.size(); i++)
    at(i)->set_configuration( config );
}

void dsp::LoadToFN::share ()
{
  MultiThread::share ();

  // TODO setup sharing of F GPU "unload"
}

void dsp::LoadToFN::finish ()
{
  MultiThread::finish ();

/*
  for (unsigned i=0; i<unloader.size(); i++)
  {
    if (Operation::verbose)
      cerr << "psr::LoadToFN::finish unloader[" << i << "]" << endl;

    unloader[i]->finish();
  }
*/
}

//! The creator of new LoadToF1 threadss
dsp::LoadToF* dsp::LoadToFN::new_thread ()
{
  return new LoadToF;
}

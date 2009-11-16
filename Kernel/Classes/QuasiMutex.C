/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "QuasiMutex.h"

//#define _DEBUG
#include "debug.h"

#include <iostream>
#include <errno.h>

QuasiMutex::QuasiMutex ()
{
  have_quit = 0;
  must_quit = 0;

  current_stream = 0;

  context = new ThreadContext;

  pthread_attr_t pat;
  pthread_attr_init (&pat);
  pthread_attr_setdetachstate (&pat, PTHREAD_CREATE_DETACHED);
  pthread_t thread_ID;

  errno = pthread_create (&thread_ID, &pat, gateway_thread, this);

  if (errno != 0)
    throw Error (FailedSys, "BatchQueue::solve", "pthread_create");
}

QuasiMutex::~QuasiMutex ()
{
  DEBUG("QuasiMutex dtor");
  delete context;
}

#include <iostream>

void QuasiMutex::quit ()
{
  DEBUG("QuasiMutex::quit lock mutex");

  ThreadContext::Lock lock (context);

  have_quit ++;
  DEBUG("QuasiMutex::quit have_quit=" << have_quit);

  context->broadcast ();
}

void QuasiMutex::add_stream (Stream* s)
{
  ThreadContext::Lock lock (context);
  stream.push_back (s);
  s->mutex = this;
  context->signal();
}

void* QuasiMutex::gateway_thread (void* thiz) try
{
  static_cast<QuasiMutex*>(thiz)->gateway();
  return 0;
}
catch (Error& error)
{
  std::cerr << "QuasiMutex::gateway_thread error" << error << std::endl;
  return 0;
}

void QuasiMutex::gateway ()
{
  ThreadContext::Lock lock (context);

  DEBUG("QuasiMutex::gateway wait until a stream has been added");

  while (stream.size() == 0)
    context->wait ();

  DEBUG("QuasiMutex::gateway stream added");

  while (stream.size())
  {
    DEBUG("QuasiMutex::gateway wait until current stream is ready");
    // (set by Stream::submit)

    while (stream.size() && stream[current_stream]->state != Stream::Ready)
    {
      context->wait ();

      unsigned remaining = must_quit - have_quit;
      if (remaining < stream.size())
	stream.resize (remaining);
    }

    if (!stream.size())
      return;

    if (stream[current_stream]->state == Stream::Ready)
    {
      DEBUG("QuasiMutex::gateway current stream is ready");

      // mark the stream as busy
      stream[current_stream]->state = Stream::Busy;

      // pure virtual queue method does the cudaMemcpyAsync, for example
      stream[current_stream]->queue ();

      current_stream ++;
    }

    DEBUG("QuasiMutex::gateway current=" << current_stream << " size=" << stream.size() << " quit=" << have_quit);

    if (current_stream == stream.size())
    {
      DEBUG("QuasiMutex::gateway launching " << stream.size() << " jobs");

      // jobs have been queued up on each stream
      current_stream = 0;

      context->unlock();

      run ();

      for (unsigned i=0; i<stream.size(); i++)
      {
	// pure virtual wait method waits for completion on stream
	stream[i]->wait ();

	// signal any threads that are waiting in Stream::join
	stream[i]->signal ();
      }

      DEBUG("QuasiMutex::gateway jobs launched");
      context->lock();
      context->broadcast();
    }
  }

  DEBUG("QuasiMutex::gateway quit");
}




//! Launch a job on one of the streams
QuasiMutex::Stream* QuasiMutex::launch (void* job)
{
  DEBUG("QuasiMutex::launch lock mutex");

  ThreadContext::Lock lock (context);

  DEBUG("QuasiMutex::launch mutex locked");

  if (stream.size() == 0)
    throw Error (InvalidState, "QuasiMutex::launch",
		 "no streams have been added to the scheduler");

  while (stream[current_stream]->state != Stream::Idle)
    context->wait ();

  DEBUG("QuasiMutex::launch current stream=" << current_stream);

  // submit the job, waking up the gateway thread
  stream[current_stream]->submit (job);

  return stream[current_stream];
}

void QuasiMutex::run ()
{
  // pure virtual run method launches data reduction on stream
  for (unsigned i=0; i<stream.size(); i++)
    stream[i]->run ();
}

QuasiMutex::Stream::Stream ()
{
  state = Idle;
  mutex = 0;
  job = 0;

  context = new ThreadContext;
}

QuasiMutex::Stream::~Stream ()
{
  delete context;
}

void QuasiMutex::Stream::submit (void* _job)
{
  if (!mutex)
    throw Error (InvalidState, "QuasiMutex::Stream::submit",
                 "parent QuasiMutex not set");

  job = _job;
  state = Ready;

  // wake up the gateway thread
  mutex->context->broadcast ();
}

void QuasiMutex::Stream::signal ()
{
  DEBUG("QuasiMutex::Stream::signal lock mutex");
  ThreadContext::Lock lock (context);
  state = Idle;

  DEBUG("QuasiMutex::Stream::signal signal");
  context->signal();

  DEBUG("QuasiMutex::Stream::signal return");
}

  
void QuasiMutex::Stream::join ()
{
  DEBUG("QuasiMutex::Stream::join lock mutex");
  ThreadContext::Lock lock (context);

  DEBUG("QuasiMutex::Stream::join wait");
  while (state != Idle)
    context->wait();

  DEBUG("QuasiMutex::Stream::join return");
};


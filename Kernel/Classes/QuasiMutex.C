/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "QuasiMutex.h"
#include <errno.h>

QuasiMutex::QuasiMutex ()
{
  context = new ThreadContext;

  pthread_attr_t pat;
  pthread_attr_init (&pat);
  pthread_attr_setdetachstate (&pat, PTHREAD_CREATE_DETACHED);
  pthread_t thread_ID;

  errno = pthread_create (&thread_ID, &pat, gateway_thread, this);

  if (errno != 0)
    throw Error (FailedSys, "BatchQueue::solve", "pthread_create");

  current_stream = 0;
}

QuasiMutex::~QuasiMutex ()
{
  delete context;
}

void QuasiMutex::add_stream (Stream* s)
{
  ThreadContext::Lock lock (context);
  stream.push_back (s);
  context->signal();
}

void* QuasiMutex::gateway_thread (void* thiz)
{
  static_cast<QuasiMutex*>(thiz)->gateway();
}

void QuasiMutex::gateway ()
{
  ThreadContext::Lock lock (context);

  // wait until a stream has been added
  while (stream.size() == 0)
    context->wait ();

  bool quit = false;
  while (!quit)
  {
    // wait until the current stream is ready (set by Stream::submit)
    while (stream[current_stream]->state != Stream::Ready)
      context->wait ();

    // mark the stream as busy
    stream[current_stream]->state == Stream::Busy;

    // pure virtual queue method does the cudaMemcpyAsync, for example
    stream[current_stream]->queue ();

    current_stream ++;

    if (current_stream == stream.size())
    {
      // jobs have been queued up on each stream

      context->unlock();

      run ();

      for (unsigned i=0; i<stream.size(); i++)
      {
	// pure virtual wait method waits for completion on stream
	stream[i]->wait ();

	// signal any threads that are waiting in Stream::join
	stream[i]->signal ();
      }

      context->lock();
    }
  }
}




//! Launch a job on one of the streams
QuasiMutex::Stream* QuasiMutex::launch (void* job)
{
  ThreadContext::Lock lock (context);

  if (stream.size() == 0)
    throw Error (InvalidState, "QuasiMutex::launch",
		 "no streams have been added to the scheduler");

  Stream* cur_stream = stream[current_stream];

  // wait for the stream to be idle
  cur_stream->join ();

  // submit the job, waking up the gateway thread
  cur_stream->submit (job);

  return cur_stream;
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
    throw Error (InvalidState, "QuasiMutex::Stream::submit");

  job = _job;
  state = Ready;

  // wake up the gateway thread
  ThreadContext::Lock lock (mutex->context);
  mutex->context->signal();
}

void QuasiMutex::Stream::signal ()
{
  ThreadContext::Lock lock (context);
  state = Idle;
  context->signal();
}

  
void QuasiMutex::Stream::join ()
{
  ThreadContext::Lock lock (context);
  while (state != Idle)
    context->wait();
};

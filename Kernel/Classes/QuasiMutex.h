//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/Attic/QuasiMutex.h,v $
   $Revision: 1.1 $
   $Date: 2009/11/14 10:46:41 $
   $Author: straten $ */

#ifndef __QuasiMutex_h
#define __QuasiMutex_h

#include "ThreadContext.h"
#include "Reference.h"

#include <vector>

//! Mutually exclusive use of a resource that performs some tasks in parallel
/*!
  A normal pthread_mutex protects a single resource, such as
  block of memory or an open file, in multi-threaded applications.
  The QuasiMutex protects a resource that can perform certain kinds of
  operations in parallel.  It was implemented with GPU processing in mind,
  where calls to cudaMemcpyAsync can be performed on multiple cudaStreams,
  but all CUDA function calls must be made from a single thread.

  A single thread acts as the gate keeper, serving any number of
  threads that queue jobs for execution.  Jobs are queued until the
  queue is full (e.g. the number of jobs equals the number of
  cudaStreams) at which point all of the jobs are launched.

  The gatekeeping thread then waits for the completion of each job,
  in order of launching, and signals any threads that are waiting on
  completion.

  In the code below, a Stream is a lot like a hyper-thread on a CPU.
  Multiple streams may operate in parallel on one resource.
*/

class QuasiMutex : public Reference::Able
{
public:
  
  class Stream;
  
  //! Default contructor launches gateway thread
  QuasiMutex ();

  //! Destructor destroys thread resources
  ~QuasiMutex ();

  //! Add a stream to the resource
  void add_stream (Stream*);
 
  unsigned get_nstream () const { return stream.size(); }
  const Stream* get_stream (unsigned i) const { return stream[i]; }
 
  //! Launch a job on one of the streams
  Stream* launch (void* job);
  
protected:

  //! Run the jobs
  virtual void run ();

  //! Mutual exclusion and condition variables used by gate keeper
  ThreadContext* context;

  //! The streams
  std::vector< Reference::To<Stream> > stream;

  //! The current stream
  unsigned current_stream;

  static void* gateway_thread (void*);
  void gateway ();

};

class QuasiMutex::Stream : public Reference::Able
{
  
protected:
  
  //! Stream states
  enum State { Idle, Ready, Busy };

  //! State of this stream
  State state;
  
  //! QuasiMutex that manages this stream
  QuasiMutex* mutex;

  //! Mutual exclusion and condition variables used for polling
  ThreadContext* context;

  //! Pointer to the job context
  void* job;

  friend class QuasiMutex;
  
  //! Submit a job (run by calling thread via Stream::launch)
  virtual void submit (void*);

  //! Send a job to the resource (run by gateway)
  virtual void queue () = 0;

  //! Run a job on the resource (run by gateway)
  virtual void run () = 0;

  //! Wait for job completion on the resource (run by gateway)
  virtual void wait () = 0;

  //! Signal job completion (run by gateway)
  virtual void signal ();

public:

  Stream ();
  ~Stream ();

  //! Wait for job completion (run directly by calling thread)
  virtual void join ();
  
};

#endif

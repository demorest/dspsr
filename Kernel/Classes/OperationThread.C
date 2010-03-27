/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/OperationThread.h"
#include <errno.h>

dsp::OperationThread::OperationThread (Operation* op)
  : Operation ( ("Thread[" + op->get_name() + "]").c_str() )
{
  the_operation = op;
  context = new ThreadContext;
  state = Idle;

  errno = pthread_create (&id, 0, operation_thread, this);

  if (errno != 0)
    throw Error (FailedSys, "dsp::OperationThread", "pthread_create");
}

void* dsp::OperationThread::operation_thread (void* ptr)
{
  reinterpret_cast<OperationThread*>( ptr )->thread ();
  return 0;
}

void dsp::OperationThread::thread ()
{
  ThreadContext::Lock lock (context);

  while (state != Quit)
  {
    while (state == Idle)
      context->wait ();

    if (state == Quit)
      return;

    the_operation->operate ();

    state = Idle;
    context->broadcast ();
  }
}

void dsp::OperationThread::prepare ()
{
  the_operation->prepare();
}

void dsp::OperationThread::reserve ()
{
  the_operation->reserve();
}

void dsp::OperationThread::add_extensions (Extensions* ext)
{
  the_operation->add_extensions (ext);
}

void dsp::OperationThread::combine (const Operation* op)
{
  const OperationThread* top = dynamic_cast<const OperationThread*>( op );
  if (top)
    op = top->the_operation;

  the_operation->combine (op);
}

void dsp::OperationThread::report () const
{
  the_operation->report ();
}

void dsp::OperationThread::reset ()
{
  the_operation->reset ();
}


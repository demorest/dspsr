//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __OperationThread_h
#define __OperationThread_h

#include "dsp/Operation.h"
#include "ThreadContext.h"

namespace dsp {

  //! Wraps an Operation in a separate thread of execution
  class OperationThread : public Operation
  {

  public:

    OperationThread (Operation*);
    ~OperationThread();
    
    void operation ();
    void prepare ();
    void reserve ();
    void add_extensions (Extensions*);
    void combine (const Operation*);
    void report () const;
    void reset ();

    class Wait;

  protected:

    static void* operation_thread (void*);
    void thread ();

    Reference::To<Operation> the_operation;
    ThreadContext* context;
    pthread_t id;

    enum State { Idle, Active, Quit };
    State state;
    
  };

  class OperationThread::Wait : public Operation
  {

  public:

    Wait (OperationThread* parent);
    void operation ();

  protected:

    Reference::To<OperationThread,false> parent;
  };
}

#endif

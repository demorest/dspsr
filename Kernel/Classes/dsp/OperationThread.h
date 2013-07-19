//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2010 - 2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __OperationThread_h
#define __OperationThread_h

#include "dsp/Operation.h"
#include "ThreadContext.h"

namespace dsp
{

  //! Executes one or more Operations in sequence in a separate thread
  /*! The Operations are performed in the order that they are added */
  class OperationThread : public Operation
  {

  public:

    //! Default constructor with optional first Operation
    OperationThread (Operation* = 0);

    //! Destructor destroys all Operation instances
    ~OperationThread();

    //! Append operation to the list of operations, thread state must be Idle
    void append_operation (Operation* op);

    //! Calls the reserve method of each Operation
    void reserve ();

    //! Calls the prepare method of each Operation
    void prepare ();

    //! Calls the add_extensions method of each Operation
    void add_extensions (Extensions* ext);

    //! Signals the operation thread to start
    void operation ();

    //! Calls the combine method of each Operation
    void combine (const Operation*);

    //! Calls the report method of each Operation
    void report () const;

    //! Calls the reset method of each Operation
    void reset ();

    //! Use this Operation to wait for completion of the operation thread
    class Wait;

    //! Return a new Wait operation for this thread
    Wait * get_wait();

    unsigned get_nop() const { return operations.size(); }
    Operation* get_op (unsigned i) { return operations.at(i); }

  protected:

    //! Operation thread calls thread method
    static void* operation_thread (void*);

    //! Calls the operation method of each Operation instance
    void thread ();

    //! The operations performed on each call to operation
    std::vector< Reference::To<Operation> > operations;

    //! Used to communicate between calling thread and operation thread
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
    OperationThread* get_parent() const { return parent; }

  protected:

    Reference::To<OperationThread,false> parent;
  };
}

#endif

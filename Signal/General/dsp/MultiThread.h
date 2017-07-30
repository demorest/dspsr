//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/MultiThread.h

#ifndef __dspsr_MultiThread_h
#define __dspsr_MultiThread_h

#include "dsp/SingleThread.h"

class ThreadContext;

namespace dsp {

  //! Multiple pipeline threads
  class MultiThread : public Pipeline
  {

  public:

    //! Constructor
    MultiThread ();
    
    //! Destructor
    ~MultiThread ();

    //! Set the number of thread to be used
    void set_nthread (unsigned);

    //! Set the configuration to be used by each thread
    void set_configuration (SingleThread::Config*);

    //! Set the Input from which data are read
    void set_input (Input*);

    //! Get the Input from which data are read
    Input* get_input ();

    //! Build the signal processing pipeline
    void construct ();

    //! Prepare to fold the input TimeSeries
    void prepare ();

    //! Set up any resources that must be shared
    virtual void share ();

    //! Run through the data
    void run ();

    //! Finish everything
    void finish ();

    //! Get the minimum number of samples required to process
    uint64_t get_minimum_samples () const;

  protected:

    //! Input
    /*! call to set_input may precede set_nthread */
    Reference::To<Input> input;

    //! Thread lock for Input::load
    ThreadContext* input_context;

    //! Condition for processing thread state changes
    ThreadContext* state_changes;

    //! The creator of new SingleThread threads
    virtual SingleThread* new_thread () = 0;

    //! The pipeline threads
    std::vector< Reference::To<SingleThread> > threads;

    //! The shared thread configuration
    Reference::To<SingleThread::Config> configuration;

    //! The thread ids
    std::vector<pthread_t> ids;

    static void* thread (void*);

    void launch_threads ();

    static void wait (SingleThread* fold, SingleThread::State st);
    static void signal (SingleThread* fold, SingleThread::State st);

  };

}

#endif // !defined(__MultiThread_h)






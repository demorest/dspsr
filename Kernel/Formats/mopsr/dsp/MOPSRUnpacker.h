/***************************************************************************
 *
 *   Copyright (C) 2013 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_MOPSRUnpacker_h
#define __dsp_MOPSRUnpacker_h

// #defined USE_UNPACK_THREADS

#include "dsp/EightBitUnpacker.h"

#ifdef USE_UNPACK_THREADS
#include "ThreadContext.h"
#endif

namespace dsp {
  
  class MOPSRUnpacker : public HistUnpacker
  {
  public:

    //! Constructor
    MOPSRUnpacker (const char* name = "MOPSRUnpacker");
    ~MOPSRUnpacker ();

    //! Cloner (calls new)
    virtual MOPSRUnpacker * clone () const;

    //! Return true if the unpacker can operate on the specified device
    bool get_device_supported (Memory*) const;

    //! Set the device on which the unpacker will operate
    void set_device (Memory*);

  protected:
    
    Reference::To<BitTable> table;

    //! Return true if we can convert the Observation
    bool matches (const Observation* observation);

    void unpack ();

    //! Return true if support the output order
    bool get_order_supported (TimeSeries::Order order) const;

    //! Set the order of the dimensions in the output TimeSeries
    virtual void set_output_order (TimeSeries::Order);

    BitSeries staging;

    void * gpu_stream;

    void unpack_on_gpu ();

    unsigned get_resolution ()const ;

  private:

    bool device_prepared;

#ifdef USE_UNPACK_THREADS
    ThreadContext * context;

    unsigned n_threads;

    unsigned thread_count;

    //! cpu_unpacker_thread ids
    std::vector <pthread_t> ids;

    //! Signals the CPU threads to start
    void start_threads ();

    //! Waits for the CPU threads to complete 
    void wait_threads ();

    //! Stops the CPU threads
    void stop_threads ();

    //! Joins the CPU threads
    void join_threads ();

    //! sk_thread calls thread method
    static void* cpu_unpacker_thread (void*);

    //! The CPU MOPSR Unpacker thread
    void thread ();

    enum State { Idle, Active, Quit };

    //! overall state
    State state;

    //! sk_thread states
    std::vector <State> states;
#endif

  };
}

#endif

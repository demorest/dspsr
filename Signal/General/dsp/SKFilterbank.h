//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/SKFilterbank.h

#ifndef __SKFilterbank_h
#define __SKFilterbank_h

#include "dsp/Filterbank.h"
#include "ThreadContext.h"

namespace dsp {
  
  //! Breaks a single-band TimeSeries into multiple frequency channels
  /*! Output will be in time, frequency, polarization order */

  class SKFilterbank: public Filterbank {

  public:

    //! Null constructor
    SKFilterbank ( unsigned _n_threads=1 );
    ~SKFilterbank ();

    //! Engine used to perform discrete convolution step
    class Engine;

    void set_engine (Engine*);

    void set_M (unsigned _M) { tscrunch = _M; }

    uint64_t get_skfb_inc (uint64_t blocksize);

    void set_output_tscr (TimeSeries * _output_tscr);

  protected:

    //! Perform the filterbank step 
    virtual void filterbank ();
    virtual void custom_prepare ();

    //! Interface to alternate processing engine (e.g. GPU)
    Reference::To<Engine> engine;

  private:

    //! number of FFTs to average to use in SK estimator
    unsigned tscrunch;

    unsigned debugd;

    //! Used to communicate between calling thread and sk_threads
    ThreadContext* context;

    //! number of threads to perform SKFilterbank operations
    unsigned n_threads;

    //! simple counter for thread identification
    unsigned thread_count;

    //! sk_thread ids
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
    static void* sk_thread (void*);

    //! The CPU SKFB thread
    void thread ();

    enum State { Idle, Active, Quit };

    //! overall state
    State state;

    //! sk_thread states
    std::vector <State> states;

    //! Tsrunched SK statistic timeseries for the current block
    Reference::To<TimeSeries> output_tscr;

    //! accumulation arrays for S1 and S2 in t scrunch
    std::vector <float> S1_tscr;
    std::vector <float> S2_tscr;

  };
 
  class SKFilterbank::Engine : public Reference::Able
  {
  public:

      virtual void setup () = 0;

      virtual void prepare (const dsp::TimeSeries* input, unsigned _nfft) = 0;

      virtual void perform (const dsp::TimeSeries* input, dsp::TimeSeries* output,
                            dsp::TimeSeries *output_tscr) = 0;

  }; 
}

#endif


//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/SingleThread.h,v $
   $Revision: 1.2 $
   $Date: 2011/08/24 22:16:11 $
   $Author: straten $ */

#ifndef __dspsr_SingleThread_h
#define __dspsr_SingleThread_h

#include "dsp/Pipeline.h"
#include "Functor.h"

class ThreadContext;

namespace dsp {

  class IOManager;
  class TimeSeries;
  class Operation;
  class Scratch;
  class Memory;

  //! A single Pipeline thread
  class SingleThread : public Pipeline
  {

  public:

    //! The MultiThread class may access private attributes
    friend class MultiThread;

    //! Stores configuration information shared between threads
    class Config;

    //! Constructor
    SingleThread ();
    
    //! Destructor
    ~SingleThread ();

    //! Set the configuration
    void set_configuration (Config*);

    //! Set the Input from which data are read
    void set_input (Input*);

    //! Get the Input from which data are read
    Input* get_input ();

    //! Prepare to fold the input TimeSeries
    void prepare ();

    //! Reserve the memory required for the pipeline
    void reserve ();

    //! Run through the data
    void run ();

    //! Share any necessary resources with the specified thread
    virtual void share (SingleThread*);

    //! Combine the results from another processing thread
    virtual void combine (const SingleThread*);

    //! Finish everything
    void finish ();

    //! Get the minimum number of samples required to process
    uint64_t get_minimum_samples () const;

    //! The verbose output stream shared by all operations
    std::ostream cerr;

    //! Take and manage a new ostream instance
    void take_ostream (std::ostream* newlog);

    unsigned thread_id;
    void set_affinity (int core);

  protected:

    //! Any special operations that must be performed at the end of data
    virtual void end_of_data ();

    //! Pointer to the ostream
    std::ostream* log;

    //! Processing thread states
    enum State
      {
	Fail,      //! an error has occurred
	Idle,      //! nothing happening
	Prepare,   //! request to prepare
	Prepared,  //! preparations completed
	Run,       //! processing started
	Done,      //! processing completed
	Joined     //! completion acknowledged 
      };

    //! Processing state
    State state;

    //! Error status
    Error error;

    //! State change communication
    ThreadContext* state_change;

    //! Mutex protecting input
    ThreadContext* input_context;

    //! Processing thread with whom sharing will occur
    SingleThread* colleague;

    //! Manages loading and unpacking
    Reference::To<IOManager> manager;

    //! The TimeSeries into which the Input is unpacked
    Reference::To<TimeSeries> unpacked;

    //! Configuration information
    Reference::To<Config> config;

    //! Create a new TimeSeries instance
    TimeSeries* new_time_series ();

    //! The operations to be performed
    std::vector< Reference::To<Operation> > operations;

    //! Insert a dump point before the named operation
    void insert_dump_point (const std::string& transformation_name);

    //! The scratch space shared by all operations
    Reference::To<Scratch> scratch;

    //! The minimum number of samples required to process
    uint64_t minimum_samples;

    Reference::To<Memory> device_memory;
    void* gpu_stream;

  };

  class SingleThread::Config : public Reference::Able
  {
  public:

    //! Default constructor
    Config ();

    //! external function used to prepare the input each time it is opened
    Functor< void(Input*) > input_prepare;

    //! report vital statistics
    bool report_vitals;

    //! report the percentage finished
    bool report_done;

    //! run repeatedly on the same input
    bool run_repeatedly;

    //! set the cuda devices to be used
    void set_cuda_device (std::string);
    unsigned get_cuda_ndevice () const { return cuda_device.size(); }

    //! set the number of CPU threads to be used
    void set_nthread (unsigned);

    //! get the total number of threads
    unsigned get_total_nthread () const;

    //! set the cpus on which each thread will run
    void set_affinity (std::string);

    //! use input-buffering to compensate for operation edge effects
    bool input_buffering;

    //! use weighted time series to flag bad data
    bool weighted_time_series;

    //! dump points
    std::vector<std::string> dump_before;

    //! get the number of buffers required to process the data
    unsigned get_nbuffers () const { return buffers; }

  protected:

    //! These attributes are set only by the SingleThread class
    friend class SingleThread;

    //! CUDA devices on which computations will take place
    std::vector<unsigned> cuda_device;

    //! CPUs on which threads will run
    std::vector<unsigned> affinity;

    //! number of CPU threads
    unsigned nthread;

    unsigned buffers;

    unsigned repeated;
  };

}

#endif // !defined(__SingleThread_h)






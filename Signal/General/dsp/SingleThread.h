//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/SingleThread.h,v $
   $Revision: 1.7 $
   $Date: 2012/01/19 21:46:17 $
   $Author: straten $ */

#ifndef __dspsr_SingleThread_h
#define __dspsr_SingleThread_h

#include "dsp/Pipeline.h"
#include "CommandLine.h"
#include "Functor.h"
#include "TextEditor.h"

class ThreadContext;

namespace dsp {

  class IOManager;
  class TimeSeries;
  class Operation;
  class Observation;
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

    //! Build the signal processing pipeline
    void construct ();

    //! Prepare the signal processing pipeline
    void prepare ();

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

    void set_input_stream (void* _input_stream) { input_stream = _input_stream; }

    // Placeholder for CUDA event signaling a completed input memory transfer
    void* input_event;

  protected:

    //! Any special operations that must be performed at the end of data
    virtual void end_of_data ();

    //! Pointer to the ostream
    std::ostream* log;

    //! Processing thread states
    enum State
      {
	Fail,        //! an error has occurred
	Idle,        //! nothing happening
	Construct,   //! request to construct
	Constructed, //! construction completed
	Prepare,     //! request to prepare
	Prepared,    //! preparations completed
	Run,         //! processing started
	Done,        //! processing completed
	Joined       //! completion acknowledged
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
    TimeSeries* new_time_series (bool increase_buffers);
    TimeSeries* new_TimeSeries () { return new_time_series(); }

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
    
    // Placeholder for CUDA stream in which input memory transfers occur
    void* input_stream;

  };

  //! Per-thread configuration options
  class SingleThread::Config : public Reference::Able
  {
  public:

    //! Default constructor
    Config ();

    //! Add command line options
    virtual void add_options (CommandLine::Menu&);

    //! Create new Input based on command line options
    Input* open (int argc, char** argv);

    //! Prepare the input according to the configuration
    virtual void prepare (Input*);

    //! external function used to prepare the input each time it is opened
    Functor< void(Input*) > input_prepare;

    // Input files represent a single continuous observation
    bool force_contiguity;

    // Command line values are header params, not file names
    bool command_line_header;

    // number of seconds to seek into data
    double seek_seconds;

    // number of seconds to process from data
    double total_seconds;

    //! List all editor-accessible attributes of the observation
    bool list_attributes;

    //! The editor used to set Observation attributes via the command line
    TextEditor<Observation> editor;

    //! report vital statistics
    bool report_vitals;

    //! report the percentage finished
    bool report_done;

    //! run repeatedly on the same input
    bool run_repeatedly;

    //! set the cuda devices to be used
    void set_cuda_device (std::string);
    unsigned get_cuda_ndevice () const { return cuda_device.size(); }

    //! set the number of kernel streams per cuda device
    void set_cuda_nstream (unsigned);
    unsigned get_cuda_nstream () const { return nstream; }

    //! set the number of CPU threads to be used
    void set_nthread (unsigned);

    //! get the total number of threads
    unsigned get_total_nthread () const;

    //! set the cpus on which each thread will run
    void set_affinity (std::string);

    //! set the FFT library
    void set_fft_library (std::string);

    //! use input-buffering to compensate for operation edge effects
    bool input_buffering;

    // keep input copies onto cuda device in their own stream so they
    // don't overlap (allows them to be faster and encourages staggered
    // kernel operations in other streams)
    bool use_input_stream;

    //! use weighted time series to flag bad data
    bool weighted_time_series;

    //! dump points
    std::vector<std::string> dump_before;

    //! get the number of buffers required to process the data
    unsigned get_nbuffers () const { return buffers; }

    //! Operate in quiet mode
    virtual void set_quiet ();

    //! Operate in verbose mode
    virtual void set_verbose ();

    //! Operate in very verbose mode
    virtual void set_very_verbose ();

  protected:

    //! These attributes are set only by the SingleThread class
    friend class SingleThread;

    //! application can make use of CUDA
    bool can_cuda;

    //! CUDA devices on which computations will take place
    std::vector<unsigned> cuda_device;

    //! number of kernel streams per cuda device
    unsigned nstream;

    //! application can make use of multiple cores
    bool can_thread;

    //! CPUs on which threads will run
    std::vector<unsigned> affinity;

    //! number of CPU threads
    unsigned nthread;

    //! number of buffers that have been created by new_time_series
    unsigned buffers;

    //! number of times that the input has been re-opened
    unsigned repeated;
  };

}

#endif // !defined(__SingleThread_h)

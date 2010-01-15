//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2007 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/LoadToFold1.h,v $
   $Revision: 1.19 $
   $Date: 2010/01/15 11:55:26 $
   $Author: straten $ */

#ifndef __baseband_dsp_LoadToFold1_h
#define __baseband_dsp_LoadToFold1_h

#include "dsp/LoadToFold.h"
#include "dsp/Filterbank.h"

#include <assert.h>

class ThreadContext;

namespace dsp {

  class IOManager;
  class TimeSeries;
  class Scratch;

  class Response;
  class RFIFilter;
  class Dedispersion;
  class ResponseProduct;

  class Operation;
  class Convolution;
  class SampleDelay;
  class PhaseLockedFilterbank;

  class Detection;
  class Fold;
  class Archiver;
  class PhaseSeriesUnloader;
  class SignalPath;

  class LoadToFoldN;

  //! A single LoadToFold thread
  class LoadToFold1 : public LoadToFold {

  public:

    //! Constructor
    LoadToFold1 ();
    
    //! Destructor
    ~LoadToFold1 ();

    //! Set the configuration to be used in prepare and run
    void set_configuration (Config*);

    //! Set the Input from which data are read
    void set_input (Input*);

    //! Get the Input from which data are read
    Input* get_input ();

    //! Prepare to fold the input TimeSeries
    void prepare ();

    //! Run through the data
    void run ();

    //! Combine the results from another processing thread
    void combine (const LoadToFold1*);

    //! Finish everything
    void finish ();

    //! Get the minimum number of samples required to process
    uint64_t get_minimum_samples () const;

    //! The verbose output stream shared by all operations
    std::ostream cerr;

    //! Take and manage a new ostream instance
    void take_ostream (std::ostream* newlog);

    unsigned thread_id;

  protected:

    friend class LoadToFoldN;

    //! Derived classes may want to do their own final preparations
    virtual void prepare_final ();

    //! Manages loading and unpacking
    Reference::To<IOManager> manager;

    //! The dedispersion kernel
    Reference::To<Dedispersion> kernel;

    //! A folding algorithm for each pulsar to be folded
    std::vector< Reference::To<Fold> > fold;
    
    //! An unloader for each pulsar to be folded
    std::vector< Reference::To<PhaseSeriesUnloader> > unloader;

    //! An unique signal path for each pulsar to be folded
    std::vector< Reference::To<SignalPath> > path;

    //! Manage the archivers
    bool manage_archiver;

    //! The RFI filter
    Reference::To<RFIFilter> rfi_filter;

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

    //! Processing thread with whom sharing will occur
    LoadToFold1* share;

  private:

    //! Configuration parameters
    Reference::To<Config> config;

    //! The TimeSeries into which the Input is unpacked
    Reference::To<TimeSeries> unpacked;

    //! Integrates the passband
    Reference::To<Response> passband;

    //! Creates the filterbank
    Reference::To<Filterbank> filterbank;

    //! Optional filterbank engine
    Reference::To<Filterbank::Engine> filterbank_engine;

    //! Performs coherent dedispersion
    Reference::To<Convolution> convolution;

    //! The product of the RFIFilter and Dedispersion kernel
    Reference::To<ResponseProduct> response_product;

    //! Removes inter-channel dispersion delays
    Reference::To<SampleDelay> sample_delay;

    //! Creates a filterbank in phase with the pulsar signal
    /*! Useful when trying to squeeze frequency resolution out of a short
      period pulsar for the purposes of scintillation measurments */
    Reference::To<PhaseLockedFilterbank> phased_filterbank;

    //! Detects the phase-coherent signal
    Reference::To<Detection> detect;

    //! Create a new TimeSeries instance
    TimeSeries* new_time_series ();

    //! The operations to be performed
    std::vector< Reference::To<Operation> > operations;

    //! Prepare to remove interchannel dispersion delays
    void prepare_interchan (TimeSeries*);

    //! Prepare to fold the given TimeSeries
    void prepare_fold (TimeSeries*);

    //! Prepare the given Archiver
    void prepare_archiver (Archiver*);

    //! The scratch space shared by all operations
    Reference::To<Scratch> scratch;

    //! The minimum number of samples required to process
    uint64_t minimum_samples;

  };

}

#endif // !defined(__LoadToFold1_h)






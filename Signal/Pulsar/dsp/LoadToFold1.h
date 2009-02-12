//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2007 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/LoadToFold1.h,v $
   $Revision: 1.12 $
   $Date: 2009/02/12 08:59:13 $
   $Author: straten $ */

#ifndef __baseband_dsp_LoadToFold1_h
#define __baseband_dsp_LoadToFold1_h

#include "dsp/LoadToFold.h"

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
  class Filterbank;
  class SampleDelay;
  class PhaseLockedFilterbank;

  class Detection;
  class Fold;
  class Archiver;
  class PhaseSeriesUnloader;

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

    //! Set the Input from which data will be read
    void set_input (Input*);

    //! Prepare to fold the input TimeSeries
    void prepare ();

    //! Run through the data
    void run ();

    //! Combine the results from another processing thread
    void combine (const LoadToFold1*);

    //! Finish everything
    void finish ();

    //! Get the minimum number of samples required to process
    uint64 get_minimum_samples () const;

    //! The verbose output stream shared by all operations
    std::ostream cerr;

    //! Take and manage a new ostream instance
    void take_ostream (std::ostream* newlog);

    unsigned id;

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

    //! Manage the archivers
    bool manage_archiver;

    //! The RFI filter
    Reference::To<RFIFilter> rfi_filter;

    //! Pointer to the ostream
    std::ostream* log;

    //! Status
    int status;

    //! Error status
    Error error;

    //! Completion notice
    ThreadContext* completion;

  private:

    //! Configuration parameters
    Reference::To<Config> config;

    //! The TimeSeries into which the Input is unpacked
    Reference::To<TimeSeries> unpacked;

    //! Integrates the passband
    Reference::To<Response> passband;

    //! Creates the filterbank
    Reference::To<Filterbank> filterbank;

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

    //! Prepared to remove interchannel dispersion delays in the given TimeSeries
    void prepare_interchan (TimeSeries*);

    //! Prepared to fold the given TimeSeries
    void prepare_fold (TimeSeries*);

    //! Prepare the given Archiver
    void prepare_archiver (Archiver*);

    //! The scratch space shared by all operations
    Reference::To<Scratch> scratch;

    //! The minimum number of samples required to process
    uint64 minimum_samples;

  };

}

#endif // !defined(__LoadToFold1_h)






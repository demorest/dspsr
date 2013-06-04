//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2012 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dspsr_LoadToF1_h
#define __dspsr_LoadToF1_h

#include "dsp/SingleThread.h"
#include "dsp/TimeSeries.h"
#include "dsp/SpatialFilterbank.h"
#include "dsp/FilterbankConfig.h"
#include "dsp/Detection.h"
#include "dsp/Accumulation.h"

#if HAVE_CUDA
#include "dsp/TransferCUDA.h"
#endif

#include "dada_hdu.h"

namespace dsp {

  //! A single LoadToF thread
  class LoadToF : public SingleThread
  {

  public:

    //! Configuration parameters
    class Config;

    //! Set the configuration to be used in prepare and run
    void set_configuration (Config*);

    //! Constructor
    LoadToF (Config* config = 0);

  private:

    //! Create the pipeline
    void construct ();

    //! Final preparations before running
    void finalize ();

    //! Configuration parameters
    Reference::To<Config> config;

    //! The filterbank in use
    Reference::To<SpatialFilterbank> filterbank;

    //! Square Law Detector
    Reference::To<Detection> detect;

    //! Accumulator / Integrator
    Reference::To<Accumulation> accumulate;

#if HAVE_CUDA
    //! GPU -> CPU Transfer
    Reference::To<TransferCUDA> copyback;
#endif

    //! Verbose output
    static bool verbose;

  };

  //! Load, unpack, process and fold data into phase-averaged profile(s)
  class LoadToF::Config : public SingleThread::Config
  {
  public:

    // Sets default values
    Config ();

    // input data block size in MB
    double block_size;

    // order in which the unpacker will output time samples
    dsp::TimeSeries::Order order;

    //! Filterbank config options
    Filterbank::Config filterbank;

    //! number of bits used to re-digitize the floating point time series
    int nbits;

    //! number of FFTs to perform in a batch
    unsigned int nbatch;

    //! number of accumulations to integrate for on detected data
    unsigned int acc_len;

    //! PSRDADA Header + Data Unit reference
    dada_hdu_t * hdu;

    //! PSRDADA shared memory key identifier
    char * hdu_key;

    //! Set quiet mode
    virtual void set_quiet ();

    //! Set verbose
    virtual void set_verbose();

    //! Set very verbose
    virtual void set_very_verbose();

  };
}

#endif // !defined(__LoadToF1_h)

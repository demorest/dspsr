//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2015 by Matthew Kerr
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// cribbed from LoadToFil

#ifndef __dspsr_LoadToFITS_h
#define __dspsr_LoadToFITS_h

#include "dsp/SingleThread.h"
#include "dsp/TimeSeries.h"
#include "dsp/Filterbank.h"
#include "dsp/FilterbankConfig.h"
#include "dsp/Dedispersion.h"
#include "dsp/OutputFile.h"

namespace dsp {

  //! A single LoadToFITS thread
  class LoadToFITS : public SingleThread
  {

  public:

    //! Configuration parameters
    class Config;

    //! Set the configuration to be used in prepare and run
    void set_configuration (Config*);

    //! Constructor
    LoadToFITS (Config* config = 0);

    //! Create the pipeline
    void construct ();

    //! Final preparations before running
    void prepare ();

  private:

    friend class LoadToFITSN;

    //! Configuration parameters
    Reference::To<Config> config;

    //! The filterbank in use
    Reference::To<Filterbank> filterbank;

    //! The dedispersion kernel
    Reference::To<Dedispersion> kernel;

    //! The output file
    Reference::To<OutputFile> outputFile;

    //! Verbose output
    static bool verbose;

  };

  //! Load, unpack, filterbank, re-digitize, and write to FITS
  class LoadToFITS::Config : public SingleThread::Config
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

    //! Maximum RAM to use (per thread)
    double maximum_RAM;

    //! dispersion measure set in output file
    double dispersion_measure;

    //! removed inter-channel dispersion delays
    bool dedisperse;

    //! coherently dedisperse along with filterbank
    bool coherent_dedisp;

    //! integrate in time before digitization
    unsigned tscrunch_factor;

    //! integrate in frequency before digitization
    unsigned fscrunch_factor;

    //! polarizations (Intensity, AABB, or Coherency)
    int npol;

    //! time interval (in seconds) between offset and scale updates
    double rescale_seconds;

    //! hold offset and scale constant after first update
    bool rescale_constant;
    
    //! number of bits used to re-digitize the floating point time series
    int nbits;

    //! integration time per output sample
    double tsamp;

    //! number of samples per output data block
    unsigned nsblk;

    //! Name of the output file
    std::string output_filename;

    //! Set quiet mode
    virtual void set_quiet ();

    //! Set verbose
    virtual void set_verbose();

    //! Set very verbose
    virtual void set_very_verbose();

  };
}

#endif // !defined(__LoadToFITS_h)






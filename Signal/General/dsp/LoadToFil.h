//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/LoadToFil.h,v $
   $Revision: 1.4 $
   $Date: 2011/12/21 06:02:20 $
   $Author: straten $ */

#ifndef __dspsr_LoadToFil_h
#define __dspsr_LoadToFil_h

#include "dsp/SingleThread.h"
#include "dsp/TimeSeries.h"
#include "dsp/Filterbank.h"
#include "dsp/FilterbankConfig.h"

namespace dsp {

  //! A single LoadToFil thread
  class LoadToFil : public SingleThread
  {

  public:

    //! Configuration parameters
    class Config;

    //! Set the configuration to be used in prepare and run
    void set_configuration (Config*);

    //! Constructor
    LoadToFil (Config* config = 0);

  private:

    //! Create the pipeline
    void construct ();

    //! Final preparations before running
    void finalize ();

    //! Configuration parameters
    Reference::To<Config> config;

    //! The filterbank in use
    Reference::To<Filterbank> filterbank;

  };

  //! Load, unpack, process and fold data into phase-averaged profile(s)
  class LoadToFil::Config : public SingleThread::Config
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

    //! time interval (in seconds) between offset and scale updates
    double rescale_seconds;

    //! hold offset and scale constant after first update
    bool rescale_constant;
    
    //! number of bits used to re-digitize the floating point time series
    int nbits;

    //! Name of the output file
    std::string output_filename;

  };
}

#endif // !defined(__LoadToFil_h)






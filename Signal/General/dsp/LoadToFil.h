//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/LoadToFil.h,v $
   $Revision: 1.2 $
   $Date: 2011/09/19 01:56:42 $
   $Author: straten $ */

#ifndef __dspsr_LoadToFil_h
#define __dspsr_LoadToFil_h

#include "dsp/SingleThread.h"
#include "dsp/TimeSeries.h"

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
    
    //! Destructor
    ~LoadToFil ();

    //! Create the pipeline
    void prepare ();

    //! Share any necessary resources with the specified thread
    // void share (SingleThread*);

    //! Run through the data
    // void run ();

    //! Finish everything
    void finish ();

  private:

    //! Configuration parameters
    Reference::To<Config> config;

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

    //! number of frequency channels in filterbank
    unsigned filterbank_nchan;

    //! dispersion measure set in output file
    double dispersion_measure;

    //! removed inter-channel dispersion delays
    bool dedisperse;

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






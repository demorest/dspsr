//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2007-2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/LoadToFold1.h,v $
   $Revision: 1.31 $
   $Date: 2011/09/19 01:56:42 $
   $Author: straten $ */

#ifndef __dspsr_LoadToFold_h
#define __dspsr_LoadToFold_h

#include "dsp/SingleThread.h"
#include "dsp/Filterbank.h"

namespace dsp {

  class TimeSeries;

  class Dedispersion;
  class Convolution;
  class Detection;
  class Fold;
  class Archiver;

  class Response;
  class RFIFilter;
  class ResponseProduct;

  class OperationThread;
  class SKFilterbank;
  class Resize;
  class SampleDelay;
  class PhaseLockedFilterbank;

  class PhaseSeriesUnloader;
  class SignalPath;

  class LoadToFoldN;

  //! A single LoadToFold thread
  class LoadToFold : public SingleThread
  {

  public:

    //! Configuration parameters
    class Config;

    //! Set the configuration to be used in prepare and run
    void set_configuration (Config*);

    //! Constructor
    LoadToFold (Config* config = 0);
    
    //! Destructor
    ~LoadToFold ();

    //! Create the pipeline
    void prepare ();

    //! Share any necessary resources with the specified thread
    void share (SingleThread*);

    //! Run through the data
    void run ();

    //! Finish everything
    void finish ();

  protected:

    friend class LoadToFoldN;

    //! Any special operations that must be performed at the end of data
    virtual void end_of_data ();

    //! Derived classes may want to do their own final preparations
    virtual void prepare_final ();

    //! Return true if the output will be divided into sub-integrations
    bool output_subints () const;

    //! The dedispersion kernel
    Reference::To<Dedispersion> kernel;

    //! A folding algorithm for each pulsar to be folded
    std::vector< Reference::To<Fold> > fold;

    //! Wrap each folder in a separate thread of execution
    std::vector< Reference::To<OperationThread> > asynch_fold;

    //! An unloader for each pulsar to be folded
    std::vector< Reference::To<PhaseSeriesUnloader> > unloader;

    //! An unique signal path for each pulsar to be folded
    std::vector< Reference::To<SignalPath> > path;

    //! Manage the archivers
    bool manage_archiver;

    //! The RFI filter
    Reference::To<RFIFilter> rfi_filter;

  private:

    //! Configuration parameters
    Reference::To<Config> config;

    //! Integrates the passband
    Reference::To<Response> passband;

    //! Optional SK filterbank
    Reference::To<SKFilterbank> skfilterbank;

    //! Optional SK Resizer 
    Reference::To<Resize> skresize;

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

    //! Prepare to remove interchannel dispersion delays
    void prepare_interchan (TimeSeries*);

    //! Prepare to fold the given TimeSeries
    void prepare_fold (TimeSeries*);

    //! Prepare the given Archiver
    void prepare_archiver (Archiver*);

  };

}

#endif // !defined(__LoadToFold_h)






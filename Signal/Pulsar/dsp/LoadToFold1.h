//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/LoadToFold1.h,v $
   $Revision: 1.1 $
   $Date: 2007/05/29 12:05:06 $
   $Author: straten $ */

#ifndef __baseband_dsp_LoadToFold1_h
#define __baseband_dsp_LoadToFold1_h

#include "dsp/LoadToFold.h"

namespace Pulsar {
  class Parameters;
  class Predictor;
}

namespace dsp {

  class IOManager;
  class TimeSeries;

  class Response;
  class RFIFilter;
  class Dedispersion;

  class Operation;
  class Convolution;
  class Filterbank;
  class SampleDelay;
  class PhaseLockedFilterbank;

  class Detection;
  class Fold;
  class Archiver;

  class LoadToFoldN;

  //! A single LoadToFold thread
  class LoadToFold1 : public LoadToFold {

  public:

    //! Constructor
    LoadToFold1 ();
    
    //! Destructor
    ~LoadToFold1 ();

    //! Set the Input from which data will be read
    void set_input (Input*);

    //! Prepare to fold the input TimeSeries
    void prepare ();

    //! Run through the data
    void run ();

  protected:

    friend class LoadToFoldN;

    //! Manages loading and unpacking
    Reference::To<IOManager> manager;

    //! The dedispersion kernel
    Reference::To<Dedispersion> kernel;

    //! A folding algorithm for each pulsar to be folded
    std::vector< Reference::To<Fold> > fold;
    
    //! An Archive writer for each pulsar to be folded
    std::vector< Reference::To<Archiver> > archiver;

    //! The RFI filter
    Reference::To<RFIFilter> rfi_filter;

    // use weighted time series
    bool weighted_time_series;
    // perform coherent dedispersion
    bool coherent_dedispersion;
    // perform coherent dedispersion while forming the filterbank
    bool simultaneous_filterbank;

    unsigned nfft;
    unsigned fres;

    // phase-locked filterbank phase bins
    unsigned plfb_nbin;
    // phase-locked filterbank channels
    unsigned plfb_nchan;

    unsigned npol;
    unsigned nbin;
    unsigned nchan;
    unsigned ndim;

    bool single_pulse;
    bool single_archive;
    double integration_length;

    // List of additional pulsar names to be folded
    std::vector<std::string> additional_pulsars;
    double reference_phase;
    double folding_period;

    std::vector< Pulsar::Parameters* > ephemerides;

    // the polynomials from which to choose a folding Pulsar::Predictor
    std::vector< Pulsar::Predictor* > predictors;

    std::string archive_class;
    std::string script;

  private:

    //! The TimeSeries into which the Input is unpacked
    Reference::To<TimeSeries> unpacked;

    //! Integrates the passband
    Reference::To<Response> passband;

    //! Creates the filterbank
    Reference::To<Filterbank> filterbank;

    //! Performs coherent dedispersion
    Reference::To<Convolution> convolution;

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

    //! Prepared to fold the given TimeSeries
    void prepare_fold (TimeSeries*);

  };

}

#endif // !defined(__LoadToFold1_h)






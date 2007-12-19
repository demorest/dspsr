//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2007 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/LoadToFoldConfig.h,v $
   $Revision: 1.3 $
   $Date: 2007/12/19 14:25:30 $
   $Author: straten $ */

#ifndef __baseband_dsp_LoadToFoldConfig_h
#define __baseband_dsp_LoadToFoldConfig_h

#include "LoadToFold.h"

namespace Pulsar {
  class Parameters;
  class Predictor;
}

namespace dsp {

  //! Load, unpack, process and fold data into phase-averaged profile(s)
  class LoadToFold::Config : public Reference::Able {

  public:

    //! Default constructor
    Config ();

    // number of time samples used to estimate undigitized power
    unsigned tbc_nsample;
    // cutoff power used for impulsive interference rejection
    float tbc_cutoff;
    // sampling threshold
    float tbc_threshold;

    // use weighted time series
    bool weighted_time_series;
    // perform coherent dedispersion
    bool coherent_dedispersion;
    // perform coherent dedispersion while forming the filterbank
    bool simultaneous_filterbank;
    // remove inter-channel dispersion delays
    bool interchan_dedispersion;

    // set the dispersion measure used in coherent dedispersion
    double dispersion_measure;

    // zap RFI during convolution
    bool zap_rfi;

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

    double reference_phase;
    double folding_period;

    /* There are three ways to fold multiple pulsars:

    1) give names: Fold will generate ephemeris and predictor
    2) give ephemeris: Fold will generate predictor
    3) give predictor: Fold will use it

    You may specify any combination of the above, but the highest numbered
    information will always be used.

    */

    // additional pulsar names to be folded
    std::vector< std::string > additional_pulsars;

    // the parameters of multiple pulsars to be folded
    std::vector< Reference::To<Pulsar::Parameters> > ephemerides;

    // the predictors of multiple pulsars to be folded
    std::vector< Reference::To<Pulsar::Predictor> > predictors;

    // name of the output archive class
    std::string archive_class;

    // name of the output archive file
    std::string archive_filename;

    // extension appended to the output archive filename
    std::string archive_extension;

    // name of post-processing psrsh script
    std::string script;

    // get the number of buffers required to process the data
    unsigned get_nbuffers () const { return buffers; }

  protected:

    // These attributes are set only by the LoadToFold classes, including
    friend class LoadToFold1;

    unsigned buffers;

  };

}

#endif

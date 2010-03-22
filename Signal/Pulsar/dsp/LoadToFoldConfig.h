//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2007 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/LoadToFoldConfig.h,v $
   $Revision: 1.24 $
   $Date: 2010/03/22 10:27:50 $
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

    // set block size to this factor times the minimum possible
    unsigned times_minimum_ndat;

    // set block size to result in approximately this much RAM usage
    uint64_t maximum_RAM;

  public:

    //! Default constructor
    Config ();

    // report vital statistics
    bool report_vitals;

    // report the percentage finished
    bool report_done;

    // set block size to this factor times the minimum possible
    void set_times_minimum_ndat (unsigned);
    unsigned get_times_minimum_ndat () const { return times_minimum_ndat; }

    // set block_size to result in approximately this much RAM usage
    void set_maximum_RAM (uint64_t);
    uint64_t get_maximum_RAM () const { return maximum_RAM; }

    // number of time samples used to estimate undigitized power
    unsigned excision_nsample;
    // cutoff power used for impulsive interference rejection
    float excision_cutoff;
    // sampling threshold
    float excision_threshold;

    // use weighted time series
    bool weighted_time_series;

    // perform coherent dedispersion
    bool coherent_dedispersion;
    // perform coherent dedispersion while forming the filterbank
    bool simultaneous_filterbank;

    // number of CUDA devices available for computation
    unsigned cuda_ndevice;

    // number of CUDA streams per device
    unsigned cuda_nstream;

    // number of threads in operation
    unsigned nthread;

    // use input-buffering to compensate for operation edge effects
    bool input_buffering;

    // remove inter-channel dispersion delays
    bool interchan_dedispersion;

    // dispersion measure used in coherent dedispersion
    double dispersion_measure;

    // zap RFI during convolution
    bool zap_rfi;

    // use FFT benchmarks to choose an optimal FFT length
    bool use_fft_bench;

    // perform phase-coherent matrix convolution (calibration)
    std::string calibrator_database_filename;

    // set fft lengths and convolution edge effect lengths
    unsigned nfft;
    unsigned nsmear;
    unsigned times_minimum_nfft;

    // phase-locked filterbank phase bins
    unsigned plfb_nbin;
    // phase-locked filterbank channels
    unsigned plfb_nchan;

    // compute and fold the fourth moments of the electric field
    bool fourth_moment;

    // compute and output mean and variance for pdmp
    bool pdmp_output;

    unsigned npol;
    unsigned nbin;
    unsigned nchan;
    unsigned ndim;

    bool force_sensible_nbin;

    bool single_pulse;
    bool single_archive;

    bool single_pulse_archives () { return single_pulse && !single_archive; }

    double integration_length;

    double reference_phase;
    double folding_period;
    bool   fractional_pulses;

    /* There are three ways to fold multiple pulsars:

    1) give names: Fold will generate ephemeris and predictor
    2) give ephemerides: Fold will generate predictors
    3) give predictors: Fold will use them

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

    // output archive post-processing jobs
    std::vector<std::string> jobs;

    // dump points
    std::vector<std::string> dump_before;

    // get the number of buffers required to process the data
    unsigned get_nbuffers () const { return buffers; }

  protected:

    // These attributes are set only by the LoadToFold classes, including
    friend class LoadToFold1;

    unsigned buffers;

  };

}

#endif

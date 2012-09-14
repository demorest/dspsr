//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2007 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/LoadToFoldConfig.h,v $
   $Revision: 1.43 $
   $Date: 2011/09/20 21:25:28 $
   $Author: straten $ */

#ifndef __baseband_dsp_LoadToFoldConfig_h
#define __baseband_dsp_LoadToFoldConfig_h

#include "dsp/LoadToFold1.h"
#include "dsp/FilterbankConfig.h"

namespace Pulsar
{
  class Parameters;
  class Predictor;
}

namespace dsp {

  //! Load, unpack, process and fold data into phase-averaged profile(s)
  class LoadToFold::Config : public SingleThread::Config
  {

    // set block size to this factor times the minimum possible
    unsigned times_minimum_ndat;

    // set block size to result in approximately this much RAM usage
    uint64_t maximum_RAM;

    // set block size to result in at least this much RAM usage
    uint64_t minimum_RAM;

  public:

    //! Default constructor
    Config ();


    // set block size to this factor times the minimum possible
    void set_times_minimum_ndat (unsigned);
    unsigned get_times_minimum_ndat () const { return times_minimum_ndat; }

    // set block_size to result in approximately this much RAM usage
    void set_maximum_RAM (uint64_t);
    uint64_t get_maximum_RAM () const { return maximum_RAM; }

    // set block_size to result in at least this much RAM usage
    void set_minimum_RAM (uint64_t);
    uint64_t get_minimum_RAM () const { return minimum_RAM; }

    // number of time samples used to estimate undigitized power
    unsigned excision_nsample;
    // cutoff power used for impulsive interference rejection
    float excision_cutoff;
    // sampling threshold
    float excision_threshold;

    // perform coherent dedispersion
    bool coherent_dedispersion;

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
    unsigned nsmear;
    unsigned times_minimum_nfft;

    // phase-locked filterbank phase bins
    unsigned plfb_nbin;
    // phase-locked filterbank channels
    unsigned plfb_nchan;

    // cyclic spectrum options
    unsigned cyclic_nchan;

    // compute and fold the fourth moments of the electric field
    bool fourth_moment;

    // compute and output mean and variance for pdmp
    bool pdmp_output;

    // apply spectral kurtosis filterbank
    bool sk_zap;

    // spectral kurtoscis integration factor
    unsigned sk_m;

    // number of stddevs to use for spectral kurtosis excision
    unsigned sk_std_devs;

    // first channel to begin SK Detection
    unsigned sk_chan_start;

    // last channel to conduct SK Detection
    unsigned sk_chan_end;

    // to disable SKDetector Fscrunch feature
    bool sk_no_fscr;

    // to disable SKDetector Tscrunch feature
    bool sk_no_tscr;

    // to disable SKDetector FT feature
    bool sk_no_ft;

    // number of CPU threads for spectral kurtosis filterbank
    unsigned sk_nthreads;

    unsigned npol;
    unsigned nbin;
    unsigned ndim;

    // Filterbank configuration options
    Filterbank::Config filterbank;

    bool force_sensible_nbin;

    bool single_pulse;
    bool single_archive;

    bool single_pulse_archives () 
    { 
      return single_pulse && !single_archive && (subints_per_archive==0); 
    }

    unsigned subints_per_archive;

    double integration_length;
    double minimum_integration_length;

    double reference_phase;
    double folding_period;
    bool   fractional_pulses;

    bool asynchronous_fold;

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

    // don't output dynamic extensions in the file
    bool no_dynamic_extensions;

    // name of the output archive class
    std::string archive_class;

    // name of the output archive file
    std::string archive_filename;

    // extension appended to the output archive filename
    std::string archive_extension;

    // output archive post-processing jobs
    std::vector<std::string> jobs;

    //! Operate in quiet mode
    virtual void set_quiet ();

    //! Operate in verbose mode
    virtual void set_verbose ();

    //! Operate in very verbose mode
    virtual void set_very_verbose ();
  };

}

#endif

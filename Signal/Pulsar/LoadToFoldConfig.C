/***************************************************************************
 *
 *   Copyright (C) 2007 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/LoadToFoldConfig.h"

using namespace std;

dsp::LoadToFold::Config::Config ()
{
  can_cuda = true;
  can_thread = true;

  minimum_RAM = 0;
  maximum_RAM = 256 * 1024 * 1024;
  times_minimum_ndat = 1;

  // number of time samples used to estimate undigitized power
  excision_nsample = 0;

  // cutoff power used for impulsive interference rejection
  excision_cutoff = -1.0;

  // sampling threshold
  excision_threshold = -1.0;

  // perform coherent dedispersion
  coherent_dedispersion = true;

  // remove inter-channel dispersion delays
  interchan_dedispersion = false;

  // over-ride the dispersion measure from the folding ephemeris
  dispersion_measure = 0.0;

  zap_rfi = false;
  use_fft_bench = false;

  // by default, dspsr will switch to TFP ordering to optimize folding
  optimal_order = true;

  times_minimum_nfft = 0;
  nsmear = 0;

  // phase-locked filterbank phase bins
  plfb_nbin = 0;
  // phase-locked filterbank channels
  plfb_nchan = 0;

  // default cyclic spectrum off
  cyclic_nchan = 0;
  // default to no oversampling
  cyclic_mover = 1;

  // do not compute the fourth order moments by default
  fourth_moment = false;

  // do not produce pdmp output by default
  pdmp_output = false;

  // do not use spectral kurtosis filterbank by default
  sk_zap = false;

  // when applying spectral kurtosis, also produce non-zapped version of output
  nosk_too = false;

  // samples to integrate to form spectral kurtosis statistic
  sk_m = 128;

  // samples to integrate to form spectral kurtosis statistic
  sk_std_devs = 3;

  // first channel to conduct spectral kurtosis detection
  sk_chan_start = 0;

  // last channel to conduct spectral kurtosis detection
  sk_chan_end = 0;

  // disables SKDetector's Fscrunch feature
  sk_no_fscr = false;

  // disables SKDetector's Tscrunch feature
  sk_no_tscr = false;

  // disables SKDetector's freq by time despeckeler
  sk_no_ft = false;

  // by default onl 1 SK thread [per CPU thread]
  sk_nthreads = 1;

  // by default, do not fold the SK filterbank output
  sk_fold = false;

  // full polarization by default
  npol = 4;

  // let Fold choose the number of bins
  nbin = 0;

  // an optimization feature used in Detection
  ndim = 4;

  // Don't allow more bins than is sensible
  force_sensible_nbin = false;

  // unload sub-integrations to separate files by default
  single_archive = false;

  // if specified, the number of sub-integrations to write to each file
  subints_per_archive = 0;

  // integrate for specified number of pulses
  integration_turns = 0;

  // integrate for specified interval length
  integration_length = 0;

  // by default, no minimum is specified
  minimum_integration_length = -1;

  // rotate the pulsar with respect to the predictor
  reference_phase = 0;

  // fold at a constant period
  folding_period = 0;

  // do not fold the fractional pulses at the beginning and end of data
  fractional_pulses = false;

  // do not fold asynchronously by default
  asynchronous_fold = false;

  // produce BasebandArchive output by default
  archive_class = "Baseband";
  archive_class_specified_by_user = false;

  // Output dynamic extensions by default
  no_dynamic_extensions = false;

}

// set block size to this factor times the minimum possible
void dsp::LoadToFold::Config::set_times_minimum_ndat (unsigned ndat)
{
  times_minimum_ndat = ndat;
  maximum_RAM = 0;
}

// set block_size to result in approximately this much RAM usage
void dsp::LoadToFold::Config::set_maximum_RAM (uint64_t ram)
{
  maximum_RAM = ram;
  minimum_RAM = 0;
  times_minimum_ndat = 1;
}

// set block_size to result in at least this much RAM usage
void dsp::LoadToFold::Config::set_minimum_RAM (uint64_t ram)
{
  minimum_RAM = ram;
  maximum_RAM = 0;
  times_minimum_ndat = 1;
}

void dsp::LoadToFold::Config::set_archive_class (const std::string& name)
{
  archive_class = name;
  archive_class_specified_by_user = true;
}

/*
  These headers are required only for setting verbosity.
*/

#include "dsp/Archiver.h"
#include "Pulsar/Archive.h"

void dsp::LoadToFold::Config::set_quiet ()
{
  SingleThread::Config::set_quiet();
  Pulsar::Archive::set_verbosity (0);
  dsp::Archiver::verbose = 0;
}

void dsp::LoadToFold::Config::set_verbose ()
{
  SingleThread::Config::set_verbose();
  Pulsar::Archive::set_verbosity (2);
  dsp::Archiver::verbose = 2;
}

void dsp::LoadToFold::Config::set_very_verbose ()
{
  SingleThread::Config::set_very_verbose();
  Pulsar::Archive::set_verbosity (3);
  dsp::Archiver::verbose = 3;
}

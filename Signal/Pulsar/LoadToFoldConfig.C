/***************************************************************************
 *
 *   Copyright (C) 2007 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/LoadToFoldConfig.h"

dsp::LoadToFold::Config::Config ()
{
  // be a little bit verbose by default
  report_done = true;
  report_vitals = true;

  // process each file once
  run_repeatedly = false;
  repeated = 0;

  minimum_RAM = 0;
  maximum_RAM = 256 * 1024 * 1024;
  times_minimum_ndat = 1;

  // number of time samples used to estimate undigitized power
  excision_nsample = 0;

  // cutoff power used for impulsive interference rejection
  excision_cutoff = -1.0;

  // sampling threshold
  excision_threshold = -1.0;

  // use weighted time series
  weighted_time_series = true;

  // perform coherent dedispersion
  coherent_dedispersion = true;

  // perform coherent dedispersion while forming the filterbank
  simultaneous_filterbank = false;

  // no CUDA devices by default
  cuda_ndevice = 0;
  // one stream per CUDA device by default
  cuda_nstream = 1;
  // one thread by default
  nthread = 1;

  // use input buffering
  input_buffering = true;

  // remove inter-channel dispersion delays
  interchan_dedispersion = false;

  // over-ride the dispersion measure from the folding ephemeris
  dispersion_measure = 0.0;

  zap_rfi = false;
  use_fft_bench = false;

  times_minimum_nfft = 0;
  nfft = 0;
  nsmear = 0;

  // phase-locked filterbank phase bins
  plfb_nbin = 0;
  // phase-locked filterbank channels
  plfb_nchan = 0;

  // do not compute the fourth order moments by default
  fourth_moment = false;

  // do not produce pdmp output by default
  pdmp_output = false;

  // full polarization by default
  npol = 4;

  // let Fold choose the number of bins
  nbin = 0;

  // no filterbank by default
  nchan = 1;

  // an optimization feature used in Detection
  ndim = 4;

  // Don't allow more bins than is sensible
  force_sensible_nbin = false;

  // full integrated profile by default
  single_pulse = false;

  // in single_pulse mode, unload integrations to separate files by default
  single_archive = false;

  // in single_pulse mode, integrate for specified time
  integration_length = 0;

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


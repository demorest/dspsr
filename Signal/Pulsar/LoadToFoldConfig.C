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
  report = 1;

  // number of time samples used to estimate undigitized power
  tbc_nsample = 0;

  // cutoff power used for impulsive interference rejection
  tbc_cutoff = 0.0;

  // sampling threshold
  tbc_threshold = 0.0;

  // use weighted time series
  weighted_time_series = true;

  // perform coherent dedispersion
  coherent_dedispersion = true;

  // perform coherent dedispersion while forming the filterbank
  simultaneous_filterbank = false;

  // remove inter-channel dispersion delays
  interchan_dedispersion = false;

  // use the dispersion measure from the folding ephemeris
  dispersion_measure = 0;

  zap_rfi = false;

  times_minimum_nfft = 0;
  nfft = 0;
  fres = 0;

  // phase-locked filterbank phase bins
  plfb_nbin = 0;
  // phase-locked filterbank channels
  plfb_nchan;

  // full polarization by default
  npol = 4;

  // let Fold choose the number of bins
  nbin = 0;

  // no filterbank by default
  nchan = 1;

  // an optimization feature used in Detection
  ndim = 4;

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

  // produce BasebandArchive output by default
  archive_class = "Baseband";

}

/***************************************************************************
 *
 *   Copyright (C) 2010 by Jonathan Khoo
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <iostream>
#include <stdlib.h>

#include "dsp/Plot.h"
#include "dsp/Input.h"
#include "dsp/Rescale.h"


using std::cerr;
using std::endl;
using std::ios;
using dsp::SampleDelay;

bool dsp::Plot::verbose = false;

dsp::Plot::Plot() :
  voltages(new dsp::TimeSeries),
  sample_delay(new dsp::SampleDelay),
  last_seconds(0.0),
  x_range(std::make_pair<float, float>(0.0, 0.0)),
  y_range(std::make_pair<float, float>(0.0, 0.0)),
  dispersion_measure(0.0),
  ndat(0),
  buffer(0.0)
{}

dsp::Plot::~Plot()
{}

void dsp::Plot::transform()
{
  if (filename.empty()) {
    throw Error(InvalidState, "dsp::Plot::transform",
        "filename not set");
  }

  Reference::To<dsp::IOManager> manager = new dsp::IOManager;
  manager->set_output(voltages);
  manager->open(filename);

  init(manager);

  // Retain the observation details for plot-specific processes.
  info = new dsp::Observation;
  info->copy(manager->get_info());

  //info->set_dispersion_measure(67);

  const unsigned npol = info->get_npol();
  const unsigned nchan = info->get_nchan();
  const uint64_t ndat = info->get_ndat();

  // XXX: think - necessary???
  //data.resize(npol);
  data.resize(1);

  // XXX: handle pol selection and stokes I (poln0 + poln1)

  data[0].resize(nchan);

  const double duration = (info->get_end_time() -
      info->get_start_time()).in_seconds();

  // Skip to the last # seconds of file if last_seconds has been specified
  if (last_seconds > 0.0) {
    x_range.second = duration - buffer;
    x_range.first = x_range.second - last_seconds;
  }

  // Plot till end of file by default
  if (x_range.second == 0.0) {
    x_range.second = duration;
  }

  // Get start and stop samples.
  const unsigned start_sample = time_as_sample(x_range.first, ndat, duration);
  const unsigned end_sample = time_as_sample(x_range.second, ndat, duration);

  // Total number of samples to be read
  int samples_remaining = end_sample - start_sample;

  // XXX: stop hardcoding everything!
  manager->set_block_size(2048);

  // Skip to first sample
  manager->get_input()->seek(start_sample);

  if (dispersion_measure != 0.0) {
    manager->get_info()->set_dispersion_measure(dispersion_measure);
  }

  do {
    manager->operate();

    if (dispersion_measure != 0.0) {
      sample_delay->set_input (voltages);
      sample_delay->set_output (voltages);
      sample_delay->set_function (new dsp::Dedispersion::SampleDelay);
      sample_delay->prepare();
      sample_delay->operate();
    }

    const unsigned ndat = voltages->get_ndat();

    //for (unsigned ipol = 0; ipol < npol; ++ipol) {
    unsigned ipol = 0;
      for (unsigned ichan = 0; ichan < nchan; ++ichan) {

        // Read ndat, unless samples_remaining < ndat
        const unsigned ndat_to_read =
          samples_remaining < ndat ? samples_remaining : ndat;

        const float* data_ptr = voltages->get_datptr(ichan, ipol);

        for (unsigned idat = 0; idat < ndat_to_read; ++idat, ++data_ptr) {
          data[ipol][ichan].push_back(*data_ptr);
        }
      }
    //}

    samples_remaining -= ndat;

  } while (samples_remaining > 0);

  // Number of time samples read.
  set_ndat(data[0][0].size());

  // Retain a copy of dsp::Observation to use for unpacking.
  info->copy(manager->get_info());
}

void dsp::Plot::set_filename(const std::string& fname)
{
  filename = fname;
}

string dsp::Plot::get_plot_title() const
{
  const string source    = voltages->get_source();
  const double frequency = voltages->get_centre_frequency();
  const double bandwidth = voltages->get_bandwidth();

  const string title = source + " " + filename +
    " Freq: " + tostring<double>(frequency, 2, ios::fixed) +
    " BW: " + tostring<double>(bandwidth, 2, ios::fixed);

  return title;
}

void dsp::Plot::set_ndat(const unsigned _ndat)
{
  ndat = _ndat;
}

unsigned dsp::Plot::get_ndat() const {
  return ndat;
}

void dsp::Plot::set_last_seconds(const float _last_seconds, float _buffer)
{
  last_seconds = _last_seconds;

  if (_buffer > 0.0) {
    buffer = _buffer;
  }
}

float dsp::Plot::get_last_seconds()
{
  return last_seconds;
}

void dsp::Plot::set_x_range(const rangeType& range)
{
  if (range.first < range.second) {
    x_range = range;
  } else if (range.first > range.second) {
    x_range.first = range.second;
    x_range.second= range.first;
  }
}

void dsp::Plot::set_y_range(const rangeType& range)
{
  if (range.first < range.second) {
    y_range = range;
  } else if (range.first > range.second) {
    y_range.first = range.second;
    y_range.second= range.first;
  }
}

uint64_t dsp::Plot::time_as_sample(const double time, const uint64_t ndat,
    const double duration)
{
  return static_cast<uint64_t>((time/duration) * ndat);
}

void dsp::Plot::set_pol(const unsigned _pol)
{
  pol = _pol;
}

float dsp::Plot::get_dispersion_measure()
{
return dispersion_measure;
}

void dsp::Plot::set_dispersion_measure(const float _dispersion_measure)
{
  dispersion_measure = _dispersion_measure;
}

/**
 * Initialise plot-specific various before performing the transformation.
 */
void dsp::Plot::init(IOManager* manager)
{
}



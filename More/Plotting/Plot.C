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
  nbit(0),
  ndat(0),
  duration(0.0),
  last_seconds(0.0),
  bandwidth(0.0),
  centre_frequency(0.0),
  x_range(std::make_pair<float, float>(0.0, 0.0)),
  y_range(std::make_pair<float, float>(0.0, 0.0)),
  dedisperse(false),
  dispersion_measure(0.0),
  buffer(0.0),
  write_summed_channels(false),
  info(new dsp::Observation)
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

  info = manager->get_info();

  //info->set_dispersion_measure(67);

  const unsigned npol = info->get_npol();
  const unsigned nchan = info->get_nchan();
  const uint64_t ndat = info->get_ndat();

  nbit = info->get_nbit();


  duration = get_duration(info->get_start_time(), info->get_end_time());

  centre_frequency = manager->get_info()->get_centre_frequency();
  bandwidth = manager->get_info()->get_bandwidth();

  // XXX: think - necessary???
  //data.resize(npol);
  data.resize(1);

  // XXX: handle pol selection and stokes I (poln0 + poln1)

  data[0].resize(nchan);

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
  const unsigned start_sample = time_as_sample(x_range.first, ndat);
  const unsigned end_sample = time_as_sample(x_range.second, ndat);

  // Total number of samples to be read
  int samples_remaining = end_sample - start_sample;

  // XXX: stop hardcoding everything!
  manager->set_block_size(2048);

  // Skip to first sample
  manager->get_input()->seek(start_sample);

  if (get_dedisperse()) {
    dedisperse_data(info);
  }

  do {
    manager->operate();

    if (get_dedisperse()) {
      sample_delay->set_input (voltages);
      sample_delay->set_output (voltages);
      sample_delay->set_function (new dsp::Dedispersion::SampleDelay);
      sample_delay->prepare();

      sample_delay->operate();
    }

    const unsigned ndat  = voltages->get_ndat();

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

uint64_t dsp::Plot::time_as_sample(const double time, const uint64_t ndat)
{
  return static_cast<uint64_t>((time/duration) * ndat);
}

void dsp::Plot::set_pol(const unsigned _pol)
{
  pol = _pol;
}

double dsp::Plot::get_duration(const MJD& start, const MJD& end)
{
  return (end - start).in_seconds();
}

/**
 * Dedisperses the frequency channels.
 */
void dsp::Plot::dedisperse_data(Observation* info)
{
  if (get_dispersion_measure() != 0.0) {
    const float dispersion = get_dispersion_measure();
    info->set_dispersion_measure(dispersion);
  }
}

float dsp::Plot::get_dispersion_measure()
{
return dispersion_measure;
}

void dsp::Plot::set_dispersion_measure(const float _dispersion_measure)
{
  dispersion_measure = _dispersion_measure;
}

void dsp::Plot::set_dedisperse(const bool _dedisperse)
{
  dedisperse = _dedisperse;
}

bool dsp::Plot::get_dedisperse()
{
  return dedisperse;
}

void dsp::Plot::set_write_summed_channels(const bool _write_summed_channels)
{
  write_summed_channels = _write_summed_channels;
}

bool dsp::Plot::get_write_summed_channels() const
{
  return write_summed_channels;
}

/**
 * Initialise plot-specific various before performing the transformation.
 */
void dsp::Plot::init(IOManager* manager)
{
}

void dsp::Plot::finalise()
{}

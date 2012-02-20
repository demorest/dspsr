#include "dsp/FrequencyVsTime.h"

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <algorithm>

#include <assert.h>

using std::cerr;
using std::cout;
using std::endl;

dsp::FrequencyVsTime::FrequencyVsTime()
{}

dsp::FrequencyVsTime::~FrequencyVsTime() {}

void dsp::FrequencyVsTime::prepare()
{
  ndat = get_ndat();

  nchan = voltages->get_nchan();
  npol = voltages->get_npol();

  if (pol >= npol) {
    throw Error(InvalidRange, "dsp::FrequencyVsTime::prepare",
        "pol to plot (%u) >= npol", pol);
  }

  values.resize(nchan * ndat);

  std::vector<float>::iterator it = values.begin();

  // XXX: handle pols correctly

  // Iterate over each pol, chan, and dat, copying the value
  // into a vector in the format of:
  //     chan0dat0, chan0dat1, ...

  //for (unsigned ipol = 0; ipol < npol; ++ipol) {
  unsigned ipol = 0;
    for (unsigned ichan = 0; ichan < nchan; ++ichan) {
      for (unsigned idat = 0; idat < ndat; ++idat, ++it) {
        //*(++it) = data[ipol][ichan][idat];
        *it = data[ipol][ichan][idat];
      }
    }
  //}

    if (dedisperse) {
      makeDedispersedChannelSums();
    }
}

void dsp::FrequencyVsTime::plot()
{

  if (dedisperse) {
    cpgsvp(0.1, 0.9, .25, 0.9);
  }

  preparePlotScale();

  // If x-axis has not being defined, plot whole file by default.
  if (x_range.first == 0 && x_range.second == 0) {
    x_range.first = 0.0;
    x_range.second = duration;
  }

  if (last_seconds != 0.0) {
    // Ensure specified last seconds doesn't exceed length of observation.
    if (last_seconds > duration) {
      last_seconds = duration;
    }

    x_range.second = duration - buffer;
    x_range.first = x_range.second - last_seconds;
  }

  const float min_value = *min_element(values.begin(), values.end());
  const float max_value = *max_element(values.begin(), values.end());
  const int minRange = int(floor(min_value));
  const int maxRange = int(ceil(max_value));

  if (y_range.first == 0 && y_range.second == 0) {
    y_range.first = centre_frequency - bandwidth/2.0;
    y_range.second = y_range.first + bandwidth;
  }

  cout << "plotting " << x_range.first << "s - " << x_range.second << "s" << endl;

  cpgswin(x_range.first, x_range.second, y_range.first, y_range.second);
  cpgbox("ABCTS",0.0,0,"ABCTSN",0.0,0);

  // Multiplicand to transform X values (0 -> ndat) into the corresponding time
  // (0 -> observation length).
  const float x_factor = (x_range.second - x_range.first) / static_cast<float>(ndat);

  // Multiplicand to transform Y values (0 -> nchan) into frequency
  // (centre - 0.5*bandwidth -> centre + 0.5*bandwidth).
  const float y_factor = bandwidth / static_cast<float>(nchan);

  // X = TR(1) + TR(2) * I + TR(3) * J
  // Y = TR(4) + TR(5) * I + TR(6) * J
  //    I: 0 -> ndat
  //    J: 0 -> nchan 
  float tr[6] = { x_range.first,
                  x_factor,
                  0.0,
                  y_range.first,
                  0.0,
                  y_factor };

  cpgwedg("RI", 2.0, 2.0, minRange, maxRange, " ");
  cpgimag(&values[0], ndat, nchan, 1, ndat, 1, nchan, minRange, maxRange, tr);

  const string title = get_plot_title();
  cpglab("", "Frequency (MHz)", title.c_str());

  if (dedisperse) {
  float max_sum = data_sums[0];
  float min_sum = data_sums[0];

  for (unsigned i = 0; i < data_sums.size(); ++i) {
    if (data_sums[i] > max_sum) {
      max_sum = data_sums[i];
    } else if (data_sums[i] < min_sum) {
      min_sum = data_sums[i];
    }
  }

  const float buffer = 0.05 * (fabs(max_sum - min_sum));

  max_sum += buffer;
  min_sum -= buffer;

  const float difference = (x_range.second - x_range.first) / ndat;
  float x_value = x_range.first;

  vector<float> x_values(ndat);
  for (unsigned i = 0; i < ndat; ++i) {
    x_values[i] = x_value;
    x_value += difference;
  }

    cpgsvp(0.1, 0.9, 0.1, .25);
    cpgswin(x_range.first, x_range.second, min_sum, max_sum);
    cpgline(ndat, &x_values[0], &data_sums[0]);
  }

  //cpgbox("ABCTSN",0.0,0,"ABCTSN",0.0,0);
  cpgbox("BCTSN",0.0,0,"ABCTSN",0.0,0);
  cpglab("Time (s)", "", "");

  cpgsch(0.75);
  cpgmtxt("B", 3.0, 1.1, 3.0, "J. Khoo");
  cpgsch(1.0);
}

void dsp::FrequencyVsTime::preparePlotScale()
{
  const float k = 20.0;
  const float x = 1.0 / k;
  float r = 0.0;
  float b  = 0.0;
  float g = 0.0;
  int j = 21;

  for (unsigned i = 0; i < k; ++i, ++j, r += x) {
    cpgscr(j, r, 0.0, 0.0);
  }

  for (unsigned i = 0; i < k; ++i, ++j, g += x) {
    cpgscr(j, 1.0, g, 0.0);
  }

  for (unsigned i = 0; i < k; ++i, ++j, b += x) {
    cpgscr(j, 1.0, 1.0, b);
  }

  cpgscir(k, j);
}

/**
 * Populate the sample_sums vector with sums across each channel for every
 * data value read.
 */
void dsp::FrequencyVsTime::makeDedispersedChannelSums()
{
  data_sums.resize(ndat);

  // Calculate sum of each channel.
  for (unsigned idat = 0; idat < ndat; ++idat) {

    float sum = 0.0;

    for (unsigned ichan = 0; ichan < nchan; ++ichan) {
      sum += data[0][ichan][idat];
    }

    data_sums[idat] = sum;
  }
}

void dsp::FrequencyVsTime::write_summed_channels_to_file()
{
  if (data_sums.empty())
    makeDedispersedChannelSums();

  std::ofstream outfile;
  outfile.open("searchplot.out");

  // Header format: # <source name> <start MJD> <centre frequency> <sample interval>
  outfile << "#"                  << "\t" <<
    info->get_source()            << "\t" <<
    info->get_start_time()        << "\t" <<
    info->get_centre_frequency()  << "\t" <<
    1.0/info->get_rate()          << "\t" <<
    endl;

  // Write summed frequency channels to searchplot.out.
  vector<float>::const_iterator it;
  for (it = data_sums.begin(); it != data_sums.end(); ++it)
    outfile << *it << endl;

  outfile.close();
}

void dsp::FrequencyVsTime::finalise()
{
  if (write_summed_channels) {
    write_summed_channels_to_file();
  }
}

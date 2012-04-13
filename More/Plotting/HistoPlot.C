#include <iostream>
#include <stdlib.h>
#include <algorithm>

#include "dsp/HistoPlot.h"

#include <assert.h>

#define MAX_BINS 258 // 2^8 + 2 (1 bin on each side)

using std::cerr;
using std::endl;
using std::string;

using std::ios;

dsp::HistoPlot::HistoPlot()
{}

dsp::HistoPlot::~HistoPlot() {}

void dsp::HistoPlot::prepare()
{
  // XXX: Input's ndat
  const unsigned ndat = get_ndat();
  const unsigned nchan = voltages->get_nchan();
  const unsigned npol = voltages->get_npol();

  if (pol >= npol) {
    throw Error(InvalidRange, "dsp::HistoPlot::prepare",
        "pol to plot (%u) >= npol", pol);
  }

  //for (unsigned ipol = 0; ipol < npol; ++ipol) {
  unsigned ipol = 0;
    for (unsigned ichan = 0; ichan < nchan; ++ichan) {
      for (unsigned idat = 0; idat < ndat; ++idat) {

        const float sample = data[ipol][ichan][idat];
        ++hist[sample];
      }
    }
  //}
}

void dsp::HistoPlot::plot()
{
  float x_vals[MAX_BINS] = {0.0};
  float y_vals[MAX_BINS] = {0.0};

  const float diff = getBinInterval();

  unsigned i;
  mapType::const_iterator it = hist.begin();

  if (!allSamplesUsed()) {
    i = 1;
    for (it = hist.begin(); it != hist.end(); ++it) {
      x_vals[i] = it->first;
      y_vals[i] = it->second;
      ++i;
    }
  } else {
    mapType::const_iterator nextIt = ++(hist.begin());
    x_vals[1] = it->first;
    y_vals[1] = it->second;
    i = 2;
    for (nextIt = ++hist.begin(); nextIt != hist.end(); ) {
      const unsigned diffFactor =
        (unsigned)((nextIt->first - it->first) / diff);

      if (diffFactor != 1) {
        i += diffFactor;
      }

      x_vals[i] = nextIt->first;
      y_vals[i] = nextIt->second;
      ++nextIt;
      ++it;
      ++i;
    }
  }

  cerr << "nbit: " << nbit << endl;

  switch (nbit) {
    case 1:
    case 2:
      x_vals[i] = x_vals[i - 1] + diff;
      x_vals[0] = x_vals[1] - 1.0;
      break;
    case 4:
    case 8:
      x_vals[i] = x_vals[i - 1] + (2 * diff);
      x_vals[0] = x_vals[1] - 2.0;
      break;
  }

  for (unsigned j = 1; j < i; ++j) {
    cerr << "Samples at: " << x_vals[j] << " = " << y_vals[j] << endl;
  }

  float max = getMaxValue();
  max += max * 0.05;

  cerr << "x_vals[0]: " << x_vals[0] << " x_vals[i]: " << x_vals[i] << " i: " << i << endl;

  cpgenv(x_vals[0], x_vals[i], 0, max, 0, 0);
  cpgbin(i + 2, x_vals, y_vals, 1);

  // calculate mean and RMS
  std::pair<float, float> mean_rms = get_mean_rms(hist);
  const float mean = mean_rms.first;
  const float rms = mean_rms.second;

  const string title = get_plot_title();
  cpglab("Digitiser Counts", "Number in Each Sample", title.c_str());

  // draw a vertical line to mark the mean
  x_vals[0] = x_vals[1] = mean;
  y_vals[0] = 0.0;
  y_vals[1] = max;
  cpgsci(2); // red
  cpgline(2, x_vals, y_vals);

  char label[100];
  sprintf(label, "mean: %.3f", mean);
  cpgtext(x_vals[1] + 0.2, y_vals[1] - y_vals[1]/20.0, label);

  // draw vertical lines on either side of the side to mark the RMS
  x_vals[0] = x_vals[1] = mean - rms;
  cpgsci(7); // yellow
  cpgline(2, x_vals, y_vals);

  x_vals[0] = x_vals[1] = mean + rms;
  cpgline(2, x_vals, y_vals);

  sprintf(label, "RMS: %.3f", rms);
  cpgtext(mean + rms + 0.2, y_vals[1]/2, label);

  cpgsci(3);

  cerr << endl;
  cerr << "mean: " << mean << endl;
  cerr << "rms: " << rms << endl;
}

std::pair<float, float> get_mean_rms(mapType& hist)
{
  float mean = 0.0, rms = 0.0;
  unsigned counter = 0;

  mapType::const_iterator it;

  for (it = hist.begin(); it != hist.end(); ++it) {
    mean += it->first * (float)it->second;
    counter += it->second;
  }

  mean /= (float)counter;

  for (it = hist.begin(); it != hist.end(); ++it) {
    rms += (float)(it->second) * pow(it->first - mean, 2);
  }

  rms /= (float)counter;
  rms = sqrt(rms);

  return std::make_pair(mean, rms);
}

float dsp::HistoPlot::getMaxValue()
{
  mapType::const_iterator it = max_element(hist.begin(), hist.end(), getMax);
  return it->second;
}

bool getMax(pairType p1, pairType p2)
{
  return p1.second < p2.second;
}


float dsp::HistoPlot::getBinInterval()
{
  mapType::const_iterator it = hist.begin();

  if (allSamplesUsed()) {
    // difference is the interval between first and second bin
    const float firstBin = it->first;
    ++it;

    return it->first - firstBin;
  } else {
    // difference is the smallest interval in the map
    mapType::const_iterator nextIt = ++(hist.begin());
    float minInterval = nextIt->first - it->first;

    for (nextIt = ++(hist.begin()); nextIt != hist.end(); ++nextIt, ++it) {
      if (nextIt->first - it->first < minInterval) {
        minInterval = nextIt->first - it->first;
      }
    }

    return minInterval;
  }
}

bool dsp::HistoPlot::allSamplesUsed()
{
  return hist.size() == (unsigned)pow(2, nbit);
}

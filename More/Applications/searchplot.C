/***************************************************************************
 *
 *   Copyright (C) 2009 by Jonathan Khoo
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "getopt.h"

#include <iostream>
#include <vector>

#include "dirutil.h"
#include "Error.h"
#include "strutil.h"

#include "dsp/Plot.h"
#include "dsp/PlotFactory.h"

const char HISTOGRAM_KEY = 'H';
const char FREQTIME_KEY = 'F';
const char* args = "D:Fg:hHKl:p:sx:y:";
enum plot_types {NO_PLOT, HISTOPLOT, FREQOPLOT};

using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::vector;

void usage()
{
  cout <<
    "A program for plotting search-mode data          \n"
    "Usage: searchplot [options] filenames            \n"
    "Configuration options:                           \n"
    " -g dev    Manually select a plot device         \n"
    "                                                 \n"
    "Selection options:                               \n"
    " -p poln   Poln to plot                          \n"
    " -x x1,x2  Zoom to this x-axis range             \n"
    " -y y1,y2  Zoom to this y-axis range             \n"
    "                                                 \n"
    "Plotting options:                                \n"
    " -H        Display a histogram of samples        \n"
    " -F        Plot frequency against time           \n"
    "                                                 \n"
    " -D <DM>   Dedisperse frequencies using DM       \n"
    " -K        Display summed channel values         \n"
    " -l sec    Plot last sec seconds of file         \n"
    "                                                 \n"
    "Output options:                                  \n"
    " -s        Write summed channels (searchplot.out)\n"
    "                                                 \n"
    "Utility options:                                 \n"
    " -h        Display this useful help page         \n";
}

static dsp::PlotFactory factory;

int main(int argc, char *argv[]) try
{
  vector<string> filenames;

  for (int i = optind; i < argc; ++i) {
    dirglob(&filenames, argv[i]);
  }

  if (filenames.size() == 0) {
    usage();
    return 0;
  }

  // Default pgplot device (xs).
  string plot_device = "/xs";

  plot_types plot_type = NO_PLOT;
  unsigned pol = 0;

  // Axes ranges.
  rangeType x_range(0.0, 0.0);
  rangeType y_range(0.0, 0.0);
  string s1, s2;
  float last_seconds = 0.0;
  bool dedisperse = false;
  float dispersion_measure = 0.0;
  bool write_summed_channels = false;

  int c;
  while ((c = getopt(argc, argv, args)) != -1) {
    switch (c) {
      case 'D':
        dispersion_measure = fromstring<float>(optarg);
        dedisperse = true;
        break;
      case 'g':
        plot_device = optarg;
        break;
      case 'h':
        usage();
        return 0;;
      case HISTOGRAM_KEY:
        plot_type = HISTOPLOT;
        break;
      case FREQTIME_KEY:
        plot_type = FREQOPLOT;
        break;
      case 'K':
        dedisperse = true;
        break;
      case 'l':
        last_seconds = fromstring<float>(optarg);
        break;
      case 'p':
        pol = fromstring<unsigned>(optarg);
        break;
      case 's':
        write_summed_channels = true;
        break;
      case 'x':
        string_split(optarg, s1, s2, ",");
        x_range = std::make_pair<float, float>(fromstring<float>(s1),
            fromstring<float>(s2));
        break;
      case 'y':
        string_split(optarg, s1, s2, ",");
        y_range = std::make_pair<float, float>(fromstring<float>(s1),
            fromstring<float>(s2));
        break;
      default:
        cerr << "invalid key" << std::endl;
        usage();
        return -1;
        break;
    }
  }

  Reference::To<dsp::Plot> plot;

  switch (plot_type) {
    case HISTOPLOT:
      plot = factory.construct("hist");
      break;
    case FREQOPLOT:
      plot = factory.construct("freq");
      break;
    case NO_PLOT:
      usage();
      return 0;
  }

  plot->set_pol(pol);
  plot->set_x_range(x_range);
  plot->set_y_range(y_range);
  plot->set_last_seconds(last_seconds);
  plot->set_dispersion_measure(dispersion_measure);
  plot->set_dedisperse(dedisperse);

  if (write_summed_channels) {
    plot->set_write_summed_channels(true);
  }

  //if (cpgopen(plot_device.c_str())) {
  if (cpgbeg(0, plot_device.c_str(), 1, 1)) {
    //throw Error(InvalidState, "searchplot - main",
        //"Could not open plot device=" + plot_device);
  }

  for (unsigned ifile = 0; ifile < filenames.size(); ++ifile) try {
    plot->set_filename(filenames[ifile]);
    plot->transform();
    plot->prepare();
    plot->plot();
    plot->finalise();
  } catch (string& error) {
    cerr << "searchplot error: " << error << std::endl;
  }

  cpgend();

  return 0;

} catch(Error& error) {
  cerr << "searchplot error: " << error << std::endl;
  return -1;
}


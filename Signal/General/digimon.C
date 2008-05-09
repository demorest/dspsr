
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// #if HAVE_CONFIG_H
// #include <config.h>
// #endif

#include "dsp/LevelMonitor.h"
#include "dsp/IOManager.h"
#include "dsp/Input.h"

#include "Error.h"

#if HAVE_PGPLOT
#include <cpgplot.h>
#endif

#include <iostream>
#include <stdlib.h>
#include <unistd.h>

using namespace std;

void usage ()
{
  cerr <<
    "digimon - monitor data, and issuing level-setting requests to stdout\n"
    "\n"
    " -c          read data consecutively (useful for debugging)\n"
    " -p          swap polarizations (useful for debugging)\n"
    " -m MBsamp   number of mega-bytes to sample\n"
    " -s sleep    number of seconds to sleep between samples blocks\n"
    " -i iter     number of iterations - defaults to convergence\n"
       << endl;
}
  
int main (int argc, char** argv) try
{
  // 64 Mpoints
  uint64 npts = 1 << 26;

  // number of iterations before quitting
  unsigned iterations = 0;

  // number of seconds to rest before making another estimate
  int rest_seconds = 30;

  // plot device
  string device = "?";

  bool swap_polarizations = false;
  bool consecutive = false;
  bool verbose = false;

  int arg = 0;

  while ((arg = getopt(argc, argv, "chm:ps:i:vV")) != -1)
  {
    switch (arg)
    {
    case 'i':
      iterations = atoi (optarg);
      break;

    case 'c':
      consecutive = true;
      break;

    case 'p':
      swap_polarizations = true;
      break;

    case 'm':
    {
      float mpts;
      sscanf (optarg, "%f", &mpts);
      npts = long(mpts * 1e6);
      break;
    }

    case 'h':
      usage ();
      return 0;

    case 's':
      rest_seconds = atoi (optarg);
      break;

    case 'V':
      dsp::Operation::verbose = true;
      dsp::Observation::verbose = true;
    case 'v':
      dsp::LevelMonitor::verbose = true;
      verbose = true;
      break;
    }
  }

  if (optind >= argc)
  {
    cerr << "digimon: please specify file name" << endl;
    return -1;
  }

  if (verbose)
    cerr << "digimon: creating IOManager" << endl;
  dsp::IOManager* manager = new dsp::IOManager;
  
  if (verbose)
    cerr << "digistat: opening file " << argv[optind] << endl;
  manager->open (argv[optind]);

  if (verbose)
    cerr << "digimon: creating LevelMonitor" << endl;
  dsp::LevelMonitor* digitizer = new dsp::LevelMonitor ();
  
  digitizer->set_input (manager);

  //digitizer->set_integration (npts);
  digitizer->set_max_iterations (iterations);
  digitizer->set_swap_polarizations (swap_polarizations);
  digitizer->set_consecutive (consecutive);
  
#if HAVE_PGPLOT
  if (cpgbeg (0, device.c_str(), 1, 1) != 1)
    {
      cerr << "digimon: error opening plot device" << endl;
      return -1;
    };
  
  cpgpap(3.0, 0.8);
#endif
  
  while (!manager->get_input()->eod())
  {
    digitizer->monitor ();

    if (rest_seconds && !manager->get_input()->eod())
    {
      if (verbose)
	cerr << "digimon: sleeping " << rest_seconds << " seconds" << endl;
      sleep (rest_seconds);
    }
  }
  
#if HAVE_PGPLOT
  cpgend ();
#endif
 
  return 0;
}
catch (Error& error) 
{
  cerr << "digimon: " << error << endl;
  return -1;
}

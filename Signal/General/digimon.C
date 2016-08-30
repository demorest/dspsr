
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/ExcisionUnpacker.h"
#include "dsp/LevelMonitor.h"
#include "dsp/IOManager.h"
#include "dsp/Input.h"

#include "Error.h"

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
    " -b fsleep   number of seconds to sleep between iterations \n"
    " -s sleep    number of seconds to sleep after levels are properly set\n"
    " -i iter     number of iterations - defaults to convergence\n"
       << endl;
}
  
int main (int argc, char** argv) try
{
  // 64 Mpoints
  uint64_t npts = 1 << 26;

  // number of iterations before quitting
  unsigned iterations = 0;

  // number of seconds to rest before making another estimate
  int rest_seconds = 30;

  // number of seconds to rest between iterations
  double between_iterations = 0.5;

  // plot device
  string device = "?";

  bool swap_polarizations = false;
  bool consecutive = false;
  bool verbose = false;

  int arg = 0;

  while ((arg = getopt(argc, argv, "cb:hm:ps:i:vV")) != -1)
  {
    switch (arg)
    {
    case 'i':
      iterations = atoi (optarg);
      break;

    case 'c':
      consecutive = true;
      break;

    case 'b':
      between_iterations = atof (optarg);
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
    cerr << "digimon: opening file " << argv[optind] << endl;
  manager->open (argv[optind]);

  //
  // disable excision
  //
  dsp::ExcisionUnpacker* excision = 0;
  excision = dynamic_cast<dsp::ExcisionUnpacker*>( manager->get_unpacker() );
  if (excision)
    excision->set_cutoff_sigma ( 0.0 );

  if (verbose)
    cerr << "digimon: creating LevelMonitor" << endl;
  dsp::LevelMonitor* monitor = new dsp::LevelMonitor ();
  
  monitor->set_input (manager);

  monitor->set_between_iterations (between_iterations);
  monitor->set_max_iterations (iterations);
  monitor->set_swap_polarizations (swap_polarizations);
  monitor->set_consecutive (consecutive);
  
  while (!manager->get_input()->eod())
  {
    monitor->monitor ();

    if (rest_seconds && !manager->get_input()->eod())
    {
      if (verbose)
	cerr << "digimon: sleeping " << rest_seconds << " seconds" << endl;
      sleep (rest_seconds);
    }
  }
  
  return 0;
}
catch (Error& error) 
{
  cerr << "digimon: " << error << endl;
  return -1;
}


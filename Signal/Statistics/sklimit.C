/***************************************************************************
 *
 *   Copyright (C) 2011 by Andrew Jameson and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "Error.h"
#include "dsp/SKLimits.h"

#include <iostream>
#include <stdlib.h>
#include <unistd.h>

using namespace std;

void usage() 
{
  cout << 
    "sklimit M\n"
    " M        number of integrations\n"
    "\n"
    " -s num   number of std deviations\n"
    " -v       verbose\n"
    " -h       print help text\n"
  << endl;

}

int main (int argc, char** argv) try
{

  unsigned M = 0;
  unsigned std_devs = 3;
  unsigned verbose = 0;

  int arg = 0;

  while ((arg=getopt(argc,argv,"hs:v")) != -1) 
  {
    switch (arg) 
    {
      case 'h':
        usage();
        return 0;

      case 's':
        std_devs = atoi(optarg);
        break;

      case 'v':
        verbose++;
        break;

      default:
        usage();
        return 0;
    }
  }

  if ((argc - optind) != 1) {
    cerr << "Error: M must be specified" << endl;
    usage();
    return EXIT_FAILURE;
  } else {
    M = atoi(argv[optind]);
  }

  if (verbose)
  {
    cerr << "M=" << M << " std_devs=" << std_devs << endl;
  }


  dsp::SKLimits limits(M, std_devs);
  limits.calc_limits();

  double from = limits.get_lower_threshold();
  double to = limits.get_upper_threshold();

  cerr << "[" << from << " - " << to << "]" << endl;

  return 0;
}

catch (Error& error)
{
  cerr << error << endl;
  return -1;
}

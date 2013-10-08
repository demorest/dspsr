/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/OptimalFFT.h"
#include <unistd.h>

using namespace std;

void usage () {}

int main(int argc, char ** argv) try
{
  unsigned nfilt = 0;
  unsigned nchan = 1;

  int c;
  while ((c = getopt(argc, argv, "hf:n:t:v")) != -1)
    switch (c) {

    case 'h':
      usage ();
      return 0;

    case 'f':
      nfilt = atoi (optarg);
      break;

    case 'n':
      nchan = atoi (optarg);
      break;

    case 't':
      FTransform::nthread = atoi (optarg);
      break;

    case 'v':
      dsp::OptimalFFT::verbose = true;
      FTransform::Bench::verbose = true;

    default:
      cerr << "invalid param '" << c << "'" << endl;
    }

  dsp::OptimalFFT kernel;

  kernel.set_nchan (nchan);
  kernel.set_simultaneous (true);
  kernel.get_nfft (nfilt);

  return 0;
}
catch (Error& error)
{
  cerr << error << endl;
  return -1;
}


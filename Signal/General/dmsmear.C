#include <iostream>
#include <unistd.h>

#include "dsp/Dedispersion.h"

using namespace std;

void usage()
{
  cerr <<
    "dmsmear - report dsp::Dedispersion parameters\n"
    "  Usage: dmsmear [-d DM] [-b BW] [-f FREQ] [-n NCHAN] -t\n"
    "\n"
    "   -d DM      dispersion measure in pc cm^-3\n"
    "   -b BW      bandwidth in MHz\n"
    "   -f FREQ    centre frequency in MHz\n"
    "   -n nchan   number of channels into which band will be divided\n"
    "   -x nfft    over-ride the optimal number of points in the transform\n"
    "   -w smear   over-ride extra fractional smearing buffer size\n"
    "\n"
    "   -q         quiet mode; print only the dispersion smearing\n"
    "   -v         enable verbosity flags in Dedispersion class\n"
    "\n"
    "  Program returns smear in microseconds as well as the minimum\n"
    "  required and optimal tranform length for coherent dedispersion\n"
       << endl;
}

int main(int argc, char ** argv)
{ try {

  bool verbose = false;
  bool quiet = false;

  float  dm = 1.0;
  double centrefreq = 1420.4;
  double bw = 20.0;
  int    nchan = 1;
  int    set_nfft = 0;

  bool   triple = false;

  int c;
  while ((c = getopt(argc, argv, "hd:b:f:n:qvw:x:")) != -1)
    switch (c) {

    case 'b':
      bw = atof (optarg);
      break;

    case 'd':
      dm = atof (optarg);
      break;

    case 'f':
      centrefreq = atof (optarg);
      break;

    case 'h':
      usage ();
      return 0;

    case 'n':
      nchan = atoi (optarg);
      break;

    case 't':
      triple = true;
      break;

    case 'q':
      quiet = true;
      break;

    case 'v':
      dsp::Shape::verbose = true;
      verbose = true;
      break;

    case 'w':
      dsp::Dedispersion::smearing_buffer = atof (optarg);
      break;

    case 'x':
      set_nfft = atoi (optarg);
      break;

    default:
      cerr << "invalid param '" << c << "'" << endl;
    }

  dsp::Dedispersion kernel;

  kernel.set_centre_frequency (centrefreq);
  kernel.set_bandwidth (bw);
  kernel.set_dispersion_measure (dm);

  kernel.set_nchan (nchan);
  kernel.set_dc_centred (triple);

  if (set_nfft)
    kernel.set_frequency_resolution (set_nfft);

  /* micro seconds */
  float smear_us = kernel.get_smearing_time () * 1e6;

  if (quiet) {
    cerr << " " << smear_us * 1e-6 << endl;
    return 0;
  }

  cerr << "\nInput Parameters:\n"
    "Centre Frequency:   " << kernel.get_centre_frequency() << " MHz\n"
    "Bandwidth:          " << kernel.get_bandwidth() << " MHz\n"
    "Dispersion Measure: " << kernel.get_dispersion_measure() << " pc/cm^3\n";

  if (nchan > 1)
    cerr <<
      "Sub-bands:          " << kernel.get_nchan() << endl;
  
  cerr << "\nOutput parameters:\n"
    "Dispersion delay:   " << kernel.delay_time() << " s\n"
    "Smearing time:      " << smear_us << " us\n";

  if (nchan > 1)  {
    smear_us = kernel.get_effective_smearing_time () * 1e6;
    cerr << "Effective Smearing: " << smear_us << " us\n";
  }

  kernel.prepare();

  unsigned nfilt = kernel.get_impulse_pos() + kernel.get_impulse_neg();
  unsigned nfft = kernel.get_minimum_ndat ();

  cerr << "\nWrap-Around: " << nfilt << " samples" << endl;
  cerr << "Minimum Kernel Length " << nfft
       << " (" << float(nfilt)/float(nfft)*100.0 << "% wrap)" << endl;

  nfft = kernel.get_ndat ();

  if (set_nfft)
    cerr << "Specified Kernel Length " << nfft;
  else
    cerr << "Optimal Kernel Length " << nfft;

  cerr << " (" << float(nfilt)/float(nfft)*100.0 << "% wrap)" << endl;

  return 0;

}
catch (...) {
  cerr << "exception thrown: " << endl;
  return -1;
}
}




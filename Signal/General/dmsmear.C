#include <iostream>
#include <unistd.h>

#include "dsp/Dedispersion.h"

void usage()
{
  cerr <<
    "dmsmear - report dsp::Dedispersion parameters\n"
    "  Usage: dmsmear [-d DM] [-b BW] [-f FREQ] [-n NCHAN] -t\n"
    "\n"
    "   -d DM      dispersion measure in pc cm^-3\n"
    "   -b BW      bandwidth in MHz\n"
    "   -f FREQ    centre frequency in MHz\n"
    "   -n NCHAN   number of channels into which band will be divided\n"
    "   -t         generate output for the triple method\n"
    "\n"
    "   -v         enable verbosity flags in Dedispersion class\n"
    "\n"
    "  Program returns smear in microseconds as well as the minimum\n"
    "  required and optimal tranform length for coherent dedispersion\n"
       << endl;
}

int main(int argc, char ** argv)
{ try {

  bool verbose = false;

  float  dm = 1.0;
  double centrefreq = 1420.4;
  double bw = 20.0;
  int    nchan = 1;

  bool   triple = false;

  int c;
  while ((c = getopt(argc, argv, "hd:b:f:n:tv")) != -1)
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

    case 'v':
      dsp::Shape::verbose = true;
      verbose = true;
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

  cerr << "\nInput Parameters:\n"
    "Centre Frequency:   " << kernel.get_centre_frequency () << " MHz\n"
    "Bandwidth:          " << kernel.get_bandwidth () << " MHz\n"
    "Dispersion Measure: " << kernel.get_dispersion_measure () << " pc/cm^3\n";

  if (nchan > 1)
    cerr <<
      "Sub-bands:          " << kernel.get_nchan() << endl;
  
  /* micro seconds */
  float smear_us = kernel.get_smearing_time () * 1e6;

  cerr << "\nOutput parameters:\n"
    "Dispersion delay:   " << kernel.delay_time() << " s\n"
    "Smearing time:      " << smear_us << " us\n";

  if (nchan > 1)  {
    smear_us = kernel.get_effective_smearing_time () * 1e6;
    cerr << "Effective Smearing: " << smear_us << " us\n";
  }

  kernel.prepare();

  cerr << "\nMinimum Kernel Length " << kernel.get_minimum_ndat () << endl;
  cerr << "Optimal Kernel Length " << kernel.get_ndat () << endl;


  return 0;

}
catch (...) {
  cerr << "exception thrown: " << endl;
  return -1;
}
}




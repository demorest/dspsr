/***************************************************************************
 *
 *   Copyright (C) 2003-2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

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
    "   -D         report on impact of dc_centred bug \n"
    "   -v         enable verbosity flags in Dedispersion class\n"
    "\n"
    "  Program returns smear in microseconds as well as the minimum\n"
    "  required and optimal tranform length for coherent dedispersion\n"
       << endl;
}

void report_dc_centred_impact (dsp::Dedispersion&);

int main(int argc, char ** argv) try
{
  bool verbose = false;
  bool quiet = false;
  bool dc_centred_report = false;

  float  dm = 1.0;
  double centrefreq = 1420.4;
  double bw = 20.0;
  int    nchan = 1;
  int    set_nfft = 0;

  bool   triple = false;

  int c;
  while ((c = getopt(argc, argv, "hDd:b:f:n:qvw:x:")) != -1)
    switch (c) {

    case 'b':
      bw = atof (optarg);
      break;

    case 'D':
      dc_centred_report = true;
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

  if (set_nfft)
    kernel.set_frequency_resolution (set_nfft);

  if (dc_centred_report)
  {
    report_dc_centred_impact (kernel);
    return 0;
  }

  /* micro seconds */
  float smear_us = kernel.get_smearing_time () * 1e6;

  if (quiet)
  {
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

  if (nchan > 1)
  {
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
catch (Error& error)
{
  cerr << error << endl;
  return -1;
}

/*
 * WvS - 25 June 2008
 * 
 * It was discovered that CPSR2 and PuMa2 file readers were setting the
 * dc_centred attribute to true by default.  This experimental parameter
 * probably should never have been introduced.  It shifts the frequencies
 * by half a channel width when creating a filterbank dispersion kernel.
 * In most cases, this is not right.
 *
 */

void report_dc_centred_impact (dsp::Dedispersion& kernel)
{
  if (kernel.get_nchan () < 2)
  {
    cerr << "dc_centred bug impacts only filterbank mode (use -n <nchan>)"
	 << endl;
    return;
  }

  kernel.prepare ();
  kernel.build ();
  kernel.set_build_delays ();

  unsigned ndat = kernel.get_ndat();
  unsigned nchan = kernel.get_nchan();

  kernel.set_dc_centred (false);
  vector<float> delays0;
  kernel.build (delays0, ndat, nchan);

  kernel.set_dc_centred (true);
  vector<float> delays1;
  kernel.build (delays1, ndat, nchan);

  assert (delays0.size() == delays1.size());
  assert (delays0.size() == ndat * nchan);

  unsigned lowest_cfreq_chan = 0;
  if (kernel.get_bandwidth() < 0.0)
    lowest_cfreq_chan = nchan - 1;

  unsigned offset = lowest_cfreq_chan * ndat;
  float delay_lo = delays1[offset] - delays0[offset];

  offset += ndat/2;
  assert (offset < ndat*nchan);
  float delay_0  = delays1[offset] - delays0[offset];

  offset += ndat/2-1;
  assert (offset < ndat*nchan);
  float delay_hi = delays1[offset] - delays0[offset];

  // the centre of each band is unaffected
  assert (delay_0 == 0);

  // cerr << ichan << " " << delay_lo << " " << delay_hi << endl;

  float smearing = fabs(delay_lo) + fabs(delay_hi);
  float assymetry = fabs(delay_lo) - fabs(delay_hi);

  cout << "Owing to dc_centred bug: \n"
    "total smearing = " << smearing << " microseconds\n"
    "assymetry = " << assymetry << " microseconds" << endl;
}


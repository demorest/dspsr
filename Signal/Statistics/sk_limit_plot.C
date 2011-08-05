/***************************************************************************
 *
 *   Copyright (C) 2011 by Andrew Jameson and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "Error.h"
#include "dsp/PearsonIV.h"
#include "dsp/SKLimits.h"

#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

#include <cpgplot.h>

using namespace std;

void usage() 
{
  cout << 
    "sk_limit_plot M\n"
    " M        number of integrations\n"
    "\n"
    " -l       disable log plot\n"
    " -s num   number of std deviations\n"
    " -x num   start x [default 0.0]\n"
    " -y num   end x [default 2.0]\n"
    " -z num   x step [default 0.1]\n"
    " -v       verbose\n"
    " -h       print help text\n"
  << endl;

}

int main (int argc, char** argv) try
{

  unsigned M = 0;

  unsigned std_devs = 3;

  unsigned verbose = 0;

  double sk_start = 0.1;

  double sk_end = 3.0;

  double sk_step = 0.05;

  char * device = "/xs";

  int arg = 0;

  int log = 1;

  int plot = 0;

  while ((arg=getopt(argc,argv,"hD:ls:vx:y:z:")) != -1) 
  {
    switch (arg) 
    {
      case 'h':
        usage();
        return 0;

      case 'D':
        device = strdup(optarg); 
        break;

      case 'l':
        log = 0;
        break;

      case 's':
        std_devs = atoi(optarg);
        break;

      case 'v':
        verbose++;
        break;

      case 'x':
        sk_start = atof(optarg);
        break;

      case 'y':
        sk_end = atof(optarg);
        break;

      case 'z':
        sk_step = atof(optarg);
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

  dsp::PearsonIV pIV(M);

  double percent_std_devs = erf((float) std_devs / sqrt(2));
  double target = (1 - percent_std_devs) / 2.0;

  if (verbose)
  {
    cerr << "sk_start=" << sk_start << " sk_end=" << sk_end << " sk_step=" << sk_step << endl;
    cerr << "M=" << M << " std_devs=" << std_devs << " percent_std_devs=" << percent_std_devs << endl;
    cerr << "target=" << target << endl;
  }

  double cf = 0;
  double ccf = 0;

  unsigned i=0;
  unsigned n_bins = (unsigned) ((sk_end - sk_start) / sk_step) + 1;

  vector<float> x_vals (n_bins);
  vector<float> cf_vals (n_bins);
  vector<float> ccf_vals (n_bins);

  double x = 0;
  double y = 0;

  // integrate to find the SK's that meet the target

  for (i=0; i<n_bins; i++)
  {
    x = sk_start + (i * sk_step);

    x_vals[i] = x;

    // compute the cumulative function, integrating the Pearson type IV pdf 
    // from -infinity to x
    cf = pIV.cf(x);

    // compute the complementary cumulative function, integrating the Pearson 
    // type IV pdf from x to infinity
    ccf = pIV.ccf(x);

    if (verbose)
      cerr << "[" << i << "] x=" << x << " cf="  << cf << " ccf= " << ccf << endl;

    cf_vals[i] = cf;
    ccf_vals[i] = ccf;
    if (log)
    {
      cf_vals[i] = log10(cf_vals[i]);
      ccf_vals[i] = log10(ccf_vals[i]);
    }
    
  }

  float xmin = 10.0;
  float xmax = 0.0;
  float ymin = 10.0;
  float ymax = 0;

  for (i=0; i<n_bins; i++)
  {
    if (finite(cf_vals[i]))
    {
      if (cf_vals[i] > ymax) 
        ymax = cf_vals[i];
      if (cf_vals[i] < ymin) 
        ymin = cf_vals[i];
    }

    if (finite(ccf_vals[i]))
    {
      if (ccf_vals[i] > ymax) 
        ymax = ccf_vals[i];
      if (ccf_vals[i] < ymin) 
        ymin = ccf_vals[i];
    }

    if (x_vals[i] > xmax)
      xmax = x_vals[i];
    if (x_vals[i] < xmin)
      xmin = x_vals[i];
  }

  ymax += 3;

  if (cpgopen(device) < 1) {
    cerr << "could not open display device " << device << endl;
    return EXIT_FAILURE;
  }

  if (verbose)
    cerr << "x:[" << xmin << "-" << xmax << "] y[" << ymin << "-" << ymax << "]" << endl;

  if (log)
    cpgenv(xmin, xmax, ymin, ymax, 0, 20);
  else
    cpgenv(xmin, xmax, ymin, ymax, 0, 0);
  //cpgswin(xmin, xmax, ymin, ymax);
  //cpgbox("BCNST", 0.0, 0, "LBCNSTV", 100.0, 10);
  cpglab("SK", "CF and CCF", "CF and CCF vs SK");

  char buffer[64];

  sprintf(buffer, "M=%d", pIV.get_M());
  cpgmtxt("T", -1.5, 0.05, 0.0, buffer);

  sprintf(buffer, "m=%f", pIV.get_m());
  cpgmtxt("T", -1.5, 0.95, 1.0, buffer);

  sprintf(buffer, "v=%f", pIV.get_v());
  cpgmtxt("T", -3.0, 0.95, 1.0, buffer);

  sprintf(buffer, "lamda=%f", pIV.get_lamda());
  cpgmtxt("T", -4.5, 0.95, 1.0, buffer);

  sprintf(buffer, "a=%f", pIV.get_a());
  cpgmtxt("T", -6.0, 0.95, 1.0, buffer);

  cpgline(n_bins, &(x_vals[0]), &(cf_vals[0]));
  cpgline(n_bins, &(x_vals[0]), &(ccf_vals[0]));

  vector<float> y_vals (2);
  x_vals[0] = xmin;
  x_vals[1] = xmax;
  y_vals[0] = target;
  y_vals[1] = target;
  if (log) 
  {
    cerr << "target=" << target << endl;
    cerr << "log10(target)=" << log10(target) << endl;
    y_vals[0] = (float) log10(target);
    y_vals[1] = (float) log10(target);
  }

  cpgsci(2);
  cpgline(2, &(x_vals[0]), &(y_vals[0]));

  // determine the vertical lines for the old style 
  y_vals[0] = ymin;
  y_vals[1] = ymax;

  float diff = std_devs * sqrt (4.0 / (float) pIV.get_M());
  cerr << "old range "<< (1 -diff) << " - " << (1+ diff) <<endl;
  cpgsci(3);

  x_vals[0] = 1 - diff;
  x_vals[1] = 1 - diff;
  cpgline(2, &(x_vals[0]), &(y_vals[0]));

  x_vals[0] = 1 + diff;
  x_vals[1] = 1 + diff;
  cpgline(2, &(x_vals[0]), &(y_vals[0]));

  cpgsci(4);

  dsp::SKLimits limits(M, std_devs);
  limits.calc_limits();
  float lower_sk = limits.get_lower_threshold();
  float upper_sk = limits.get_upper_threshold();
  cerr << "new range "<< lower_sk << " - " << upper_sk <<endl;

  x_vals[0] = lower_sk;
  x_vals[1] = lower_sk;
  cpgline(2, &(x_vals[0]), &(y_vals[0]));

  x_vals[0] = upper_sk;
  x_vals[1] = upper_sk;
  cpgline(2, &(x_vals[0]), &(y_vals[0]));


  cpgclos();

  return 0;
}

catch (Error& error)
{
  cerr << error << endl;
  return -1;
}

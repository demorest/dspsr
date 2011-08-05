/***************************************************************************
 *
 *   Copyright (C) 2011 by Andrew Jameson and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "Error.h"
#include "dsp/PearsonIV.h"

#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <float.h>
#include <cpgplot.h>

using namespace std;

void usage() 
{
  cout << 
    "sk_distrib_plotM\n"
    " M        number of integrations\n"
    "\n"
    " -D dev   use pgplot device\n"
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

  unsigned verbose = 0;

  double sk_start = 0.01;

  double sk_end = 3.0;

  double sk_step = 0.05;

  char * device = "/xs";

  int arg = 0;

  int plot = 0;

  while ((arg=getopt(argc,argv,"hD:vx:y:z:")) != -1) 
  {
    switch (arg) 
    {
      case 'h':
        usage();
        return 0;

      case 'D':
        device = strdup(optarg);
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

  if (verbose)
  {
    cerr << "sk_start=" << sk_start << " sk_end=" << sk_end << " sk_step=" << sk_step << endl;
    cerr << "M=" << M << endl;
  }

  unsigned i=0;
  unsigned n_bins = (unsigned) ((sk_end - sk_start) / sk_step) + 1;

  vector<float> x_vals (n_bins);
  vector<float> y_vals (n_bins);

  double x = 0;
  double y = 0;

  if (verbose > 1)
    cerr << "nbins=" << n_bins << endl;

  for (i=0; i<n_bins; i++)
  {
    x = sk_start + (i * sk_step);
    y = pIV(x);

    x_vals[i] = (float) x;
    if (finite(y))
      y_vals[i] = (float) log(y);
    else
      y_vals[i] = -1 * FLT_MAX;

    if (verbose > 1)
      cerr << "[" << i << "] x=" << x << " y=" << y << " ploty=" << y_vals[i] << endl;
  }

  float xmin = 10000000.0;
  float xmax = 0.0;
  float ymin = 10000000.0;
  float ymax = 0;

  for (i=0; i<n_bins; i++)
  {
    if (finite(y_vals[i]))
    {
      if (y_vals[i] > ymax)
        ymax = y_vals[i];
      if (y_vals[i] < ymin)
        ymin = y_vals[i];
    }

    if (x_vals[i] > xmax)
      xmax = x_vals[i];
    if (x_vals[i] < xmin)
      xmin = x_vals[i];
  }
 
  ymax += 1;
  if (ymin < -20)
    ymin = -20;

  if (cpgopen(device) < 1) {
    cerr << "could not open display device " << device << endl;
    return EXIT_FAILURE;
  }

  if (verbose)
    cerr << "x:[" << xmin << "-" << xmax << "] y[" << ymin << "-" << ymax << "]" << endl;

  cpgenv(xmin, xmax, ymin, ymax, 0, 20);
  cpglab("SK", "Probability", "Probability vs SK");

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

  cpgline(n_bins, &(x_vals[0]), &(y_vals[0]));

  x_vals[0] = 1.0;
  x_vals[1] = 1.0;
  y_vals[0] = ymin;
  y_vals[1] = ymax;

  cpgsci(2);
  cpgline(2, &(x_vals[0]), &(y_vals[0]));

  cpgclos();

  return 0;
}

catch (Error& error)
{
  cerr << error << endl;
  return -1;
}

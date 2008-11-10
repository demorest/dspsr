/***************************************************************************
 *
 *   Copyright (C) 2008 by Ramesh Bhat
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "median_smooth.h"

#include <assert.h>
#include <iostream>
#include <unistd.h>

#include "dsp/Dedispersion.h"

using namespace std;
using namespace fft;

void usage()
{
  cerr <<
    "\n"
    "  cbird - check bandpass for birdies and output a list\n"
    "\n"
    "  Usage: cbird [-b inputfile] [-t threshold] [-w window] [-o outputfile]\n"
    "\n"
    "   -b inputfile    input bandpass file from the_decimator\n"
    "   -t threshold    threshold in units of sigma (def: 4.0)\n"
    "   -w window       window for median filtering (def: 0.01)\n"
    "   -o outputfile   output file - a list of birdie channels\n"
    "\n"
    "   -q         quiet mode; print only the dispersion smearing\n"
    "   -v         enable verbosity flags \n"
    "\n"
    "  Program runs a median filter on bandpass to produce a list of birdies\n" 
       << endl;
}

int main(int argc, char ** argv) try
{
  bool verbose = false;
  bool quiet = false;

  float  thld = 4.0, window = 0.01;
  FILE*  infile = stdin;
  FILE*  outfile = stdout;
  char*  bandfile = 0;
  char*  listfile = 0;

  double centrefreq = 1420.4;
  double bw = 20.0;
  int    set_nfft = 0;


  bool   triple = false;

  int c;

  if (argc == 1) {
    cerr <<
    "\n"
    << "  Please specify an input filename " << endl;
    usage();
    exit(1);
  }

  while ((c = getopt(argc, argv, "hb:t:w:o:qv:")) != -1)
    switch (c) {

    case 'b':
      bandfile = optarg;
      break;

    case 't':
      thld = atof (optarg);
      break;

    case 'w':
      window = atof (optarg);
      break;

    case 'o':
      listfile = optarg;
      break;

    case 'h':
      usage ();
      return 0;

    case 'q':
      quiet = true;
      break;

    case 'v':
      verbose = true;
      break;

    default:
      cerr << "Invalid param '" << c << "'" << endl;
    }

// print the input parameters
  cout << "\n";
  cout << "Input file : " << bandfile << endl;
  cout << "Threshold  : " << thld << endl;
  cout << "Window     : " << window << endl;
  cout << "Output file: " << listfile << endl;
  cout << "\n";

// read in data 
  float *bpdata;
  unsigned i, ndat, nchan = 1024;
  unsigned nread, block = 1024;
  string filename = bandfile;


  bpdata = (float *) malloc(nchan*sizeof(float));

  infile = fopen(filename.c_str(), "rb");
  if (!infile)
      	throw Error (FailedSys, "",
                   "Could not open " + filename + " for input");

  for (i=0; i < nchan; i++) {
  	nread = fread(&bpdata[i],1,sizeof(float),infile);
        // cout << bpdata[i] << endl;
  }

// median filtering
  ndat = nchan;
  vector<float> data (ndat);
  vector<float> mask (ndat);

  cout << "Data size (number of channels): " << data.size() << endl;

  unsigned window_size = data.size() * window; // default is a 1% duty cycle
  float cutoff = thld; // default is a four sigma threshold

  for (i=0; i < nchan; i++) data[i] = bpdata[i];

  median_filter (data, window_size, cutoff);

  if (verbose) {
      for (i=0; i < nchan; i++)
        cout << data[i] << endl;
  }

  for (i=0; i < nchan; i++) {
     	if (data[i] != 0.0) mask[i] = 1.0;
	else mask[i] = data[i];
  }

  if (verbose) {
      for (i=0; i < nchan; i++)
        cout << mask[i] << endl;
  }

// write out channel masks
  filename = listfile;

  outfile = fopen(filename.c_str(), "w");
  if (!outfile)
      	throw Error (FailedSys, "",
                   "Could not open " + filename + " for output");

  for (i=0; i < nchan; i++)
        fprintf(outfile, "%f\n", mask[i]);

// close files
  fclose (infile);
  fclose (outfile);

  cout << "Channel masks are written into: " << filename << endl;
  cerr << "\n";

// write out input band and filtered band in ascii
  string plotfile = filename + ".band";

  outfile = fopen(plotfile.c_str(), "w");
  if (!outfile)
      	throw Error (FailedSys, "",
                   "Could not open " + filename + " for output");

  for (i=0; i < nchan; i++)
        fprintf(outfile, "%d %f %f\n", i+1, bpdata[i], data[i]);

  fclose (outfile);


// free memory
  free (bpdata);
  return 0;

}
catch (Error& error)
{
  cerr << error << endl;
  return -1;
}


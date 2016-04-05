#ifndef _FITS_PARAMS_H
#define _FITS_PARAMS_H

#define MAX_CHARS 180
#define MAX_SUBINTS 1024

struct fits_params
{
  MJD start_time;
  long day;
  long sec;
  double frac;

  int nsubint;
  int nrow;
  int nsuboffs;

  double tsamp;

  int signint;
  float zero_off;
};

#endif

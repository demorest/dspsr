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
  double tsamp;

  // The offset applied to each value to form a zero-centred mean.
  int zero_off;

  // The signed-ness of the data.
  bool is_unsigned;
};

#endif

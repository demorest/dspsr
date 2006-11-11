/***************************************************************************
 *
 *   Copyright (C) 1999 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "MJD.h"
#include "Error.h"

#include "quadruple.h"
#include "ieee.h"

#ifdef sun

int parse_quadruple (long double mjd, double* ndays, double* seconds, 
		    double* fracseconds)
{
  unsigned long intdays = (unsigned long) mjd;
  long double back_again = (long double) intdays;
  if (back_again > mjd) {
    back_again  -= 1.0;
  }
  *ndays = (double) back_again;

  /* calculate number of seconds left */
  mjd = (mjd - back_again) * 86400;
  unsigned long intseconds = (unsigned long) mjd;
  back_again = (long double) intseconds;
  if (back_again > mjd) {
    back_again  -= 1.0;
  }
  *seconds = (double) back_again;

  /* and fractional seconds left */
  *fracseconds = (double) (mjd - back_again);
  return 0;
}

MJD mjd (quadruple mjd)
{
  if (verbose)
    cerr << "MJD (quadruple)" << endl;
  double ndays = 0.0, seconds = 0.0, fracseconds = 0.0;
  parse_quadruple ((long double)mjd, &ndays, &seconds, &fracseconds);
  *this = MJD (ndays,seconds,fracseconds);
}

#else

MJD mjd (quadruple mjd)
{
  double ndays = 0.0, fracdays = 0.0;

  /* Stuart's ieee.C function */
  if (cnvrt_long_double ((u_char*) &mjd, &ndays, &fracdays) < 0)
    throw Error (FailedCall, "MJD(quadruple)", "error cnvrt_long_double");
  
  return MJD ((int)ndays, fracdays);
}

#endif

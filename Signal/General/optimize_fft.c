/***************************************************************************
 *
 *   Copyright (C) 1998 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "environ.h"

#include <stdio.h>
#include <math.h>

uint64_t power_of_two (uint64_t number)
{
  uint64_t twos = 1;
  while (twos < number)
    twos *= 2;
  if (twos != number)
    return 0;
  return 1;
}

/* ***********************************************************************
   optimal_fft_length - calculates the optimal FFT length to use when 
                        performing multiple FFTs on a long time series.
			Assumes that FFT is an O(NlogN) operation.

   GIVEN:
   nbadperfft - points thrown away from each FFT
   nfft_max   - maximum FFT length possible (due to ram limitations)
   verbose    - how noisy?
   
   RETURNS:
   The return value is the optimal FFT length, nfft, or -1 on error.

   -----------------------------------------------------------------------

   Where ngood = nfft - nbadperfft, the following ideas are held:
   
   1) performance is better if ngood is a large fraction of nfft,
   2) FFT performance is much better with smaller nfft.

   Algebraically speaking, the timescale for one FFT is proportional
   to NlogN, or:
                  t = nfft * log(nfft)

   The number of FFTs performed on a segment of M time samples is:

                 Nf = M / (nfft - nbadperfft)

   The total time spent on FFTs is then:

                  T = t * Nf = nfft * log(nfft) * M/(nfft-nbadperfft)

   Where M may be considered constant, and nbadperfft is given,
   this function aims to minimize:

            T(nfft) = nfft * log(nfft) / (nfft-nbadperfft)

   Written by Willem van Straten Sep 17, 1999
   *********************************************************************** */

uint64_t optimal_fft_length (uint64_t nbadperfft, uint64_t nfft_max, char verbose)
{  
  uint64_t nfft_min = 0;   /* the smallest FFT possible, given nbadperfft */
  uint64_t nfft = 0;       /* return value */

  double data_kept = 0.0;  /* fractional data kept from each nfft */
  double order_fft = 0.0;  /* order NlogN assumed... */
  double timescale = 0.0;  /* T(nfft) given above */
  double prev_timescale;   /* place holder of previous test result */
  
  if (!nbadperfft)  {
    fprintf (stderr, "optimal_fft_length: nbadperfft == 0\n");
    fprintf (stderr, "optimal_fft_length: give nfft to 'size_job()'\n");
    return -1;
  }

  /* find the next power of two up from nbadperfft */
  nfft_min = (uint64_t) pow (2.0, ceil(log(nbadperfft)/log(2.0))); 
  
  if (verbose || (nfft_max && nfft_max < nfft_min)) {
    fprintf(stderr,"optimal_fft_length: minimum FFT:"UI64" maximum FFT:"UI64"\n", 
	    nfft_min, nfft_max);
    fflush(stdout);
  }

  if (nfft_max && nfft_max < nfft_min)  {
    fprintf (stderr, "optimal_fft_length: ERROR max FFT < min FFT\n");
    return -1;
  }
  
  /* calculate the timescale factor */
  nfft = nfft_min;

  order_fft = (double)nfft * log (nfft);
  timescale = order_fft / (double)(nfft - nbadperfft);

  if (verbose) {
    data_kept = (double)(nfft-nbadperfft) / (double)nfft;
    fprintf (stderr,
	     "NFFT "UI64"  %%kept:%6.4f O(FFT):%6.4f Timescale:%6.4f\n", 
	     nfft, data_kept, order_fft, timescale);
  }

  while (nfft_max == 0 || nfft * 2 < nfft_max)  {

    prev_timescale = timescale;
    nfft *= 2;

    order_fft = (double) nfft * log (nfft);
    timescale = order_fft / (double)(nfft - nbadperfft);

    if (verbose) {
      data_kept = (double)(nfft-nbadperfft) / (double)nfft;
      fprintf (stderr,
	       "NFFT "UI64"  %%kept:%6.4f O(FFT):%6.4f Timescale:%6.4f\n", 
	       nfft, data_kept, order_fft, timescale);
    }

    if (timescale > prev_timescale) {
      nfft /= 2;
      break;
    }
  }
  return nfft;
}


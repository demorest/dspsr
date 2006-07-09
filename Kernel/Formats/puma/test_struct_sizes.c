/***************************************************************************
 *
 *   Copyright (C) 2003 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include <stdio.h>

#if !defined(MALIGN_DOUBLE)
#define NO_MALIGN_DOUBLE 1
#endif

#include "libpuma.h"

int main ()
{
  char* ptr1 = 0;
  char* ptr2 = 0;

  General_type general;
  Mode_type mode;

  if (sizeof (Telescope_type) !=   144) {
    fprintf (stderr, "sizeof (Telescope) = %d != 32\n",
             sizeof (Telescope_type));
    return -1;
  }

  if (sizeof (Band_type) !=    32) {
    fprintf (stderr, "sizeof (Band) = %d != 32\n",
             sizeof (Band_type));
    return -1;
  }
  
  ptr1 = (char*) &general;
  ptr2 = (char*) &(general.DataMJD);
  fprintf (stderr, "General_type.DataMJD = %d\n", ptr2-ptr1);
  
  ptr2 = (char*) &(general.DataTime);
  fprintf (stderr, "General_type.DataTime = %d\n", ptr2-ptr1);

  if ( (ptr2-ptr1) % 8 )
    fprintf (stderr, "double General_type.DataTime is not aligned\n");

  if (sizeof (General_type) !=   488) {
    fprintf (stderr, "sizeof (General) = %d != 488\n",
             sizeof (General_type));
    return -1;
  }
  
  if (sizeof (Observatory_type) !=    40) {
    fprintf (stderr, "sizeof (Observatory) = %d != 40\n",
             sizeof (Observatory_type));
    return -1;
  }
  
  if (sizeof (Observation_type) !=   248) {
    fprintf (stderr, "sizeof (Observation) = %d != 248\n",
             sizeof (Observation_type));
    return -1;
  }
  
  if (sizeof (Target_type) !=    48) {
    fprintf (stderr, "sizeof (Target) = %d != 48\n", sizeof (Target_type));
    return -1;
  }
  
  if (sizeof (Signalpath_type) !=  2344) {
    fprintf (stderr, "sizeof (Signalpath) = %d != 2344\n",
             sizeof (Signalpath_type));
    return -1;
  }
  
  ptr1 = (char*) &mode;

  ptr2 = (char*) &(mode.FIRFactor);
  fprintf (stderr, "Mode_type.FIRFactor = %d\n", ptr2-ptr1);

  ptr2 = (char*) &(mode.NDMs);
  fprintf (stderr, "Mode_type.NDMs = %d\n", ptr2-ptr1);

  ptr2 = (char*) &(mode.DM[0]);
  fprintf (stderr, "Mode_type.DM[0] = %d\n", ptr2-ptr1);

  if ( (ptr2-ptr1) % 8 )
    fprintf (stderr, "double Mode_type.DM is not aligned\n");

  ptr2 = (char*) &(mode.ScaleDynamic);
  fprintf (stderr, "Mode_type.ScaleDynamic = %d\n", ptr2-ptr1);

  ptr2 = (char*) &(mode.AdjustInterval);
  fprintf (stderr, "Mode_type.AdjustInterval = %d\n", ptr2-ptr1);

  if ( (ptr2-ptr1) % 8 )
    fprintf (stderr, "double Mode_type.AdjustInterval is not aligned\n");

  if (sizeof (Mode_type) !=   264) {
    fprintf (stderr, "sizeof (Mode) = %d != 264\n", sizeof (Mode_type));
    return -1;
  }
  
  if (sizeof (Software_type) !=   512) {
    fprintf (stderr, "sizeof (Software) = %d != 512\n",
             sizeof (Software_type));
    return -1;
  }
  
  if (sizeof (Check_type) !=    64) {
    fprintf (stderr, "sizeof (Check) = %d != 64\n", sizeof (Check_type));
    return -1;
  }
  
  if (sizeof (Reduction_type) !=   496) {
    fprintf (stderr, "sizeof (Reduction) = %d != 496\n",
             sizeof (Reduction_type));
    return -1;
  }
  
  if (sizeof (Header_type) !=  4504) {
    fprintf (stderr, "sizeof (Header) = %d != 4504\n", sizeof (Header_type));
    return -1;
  }
  
  return 0;
  
}


/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include <stdio.h>

#include <inttypes.h>

#include "environ.h"

int main ()
{
  int64_t test_int64_t = -987654321;
  uint64_t test_uint64_t = 0;
  char buffer[256];

  sprintf (buffer, I64, test_int64_t);

  if (strcmp(buffer, "-987654321") != 0) {
    fprintf (stderr, "Error printing signed 64-bit integer\n");
    return -1;
  }

  sscanf ("1234567890", UI64, &test_uint64_t);

  if (test_uint64_t != 1234567890) {
    fprintf (stderr, "Error parsing unsigned 64-bit integer\n");
    return -1;
  }

  
  return 0;
}


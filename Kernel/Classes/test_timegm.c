/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <stdio.h>
#include <time.h>

int main ()
{
  const char* buffer = "2010-04-12-14:37:56";

  struct tm utc;
  if (strptime (buffer, "%Y-%m-%d-%H:%M:%S", &utc) == NULL)
  {
    fprintf (stderr, "failed strptime (%s)\n", buffer);
    return -1;
  }

  fprintf (stderr, "calling timegm\n");

  time_t temp = timegm (&utc);

  printf ("timegm returns %s\n", ctime(&temp));

  return 0;
}

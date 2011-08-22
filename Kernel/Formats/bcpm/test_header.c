
#include "dsp/bpphdr.h"
#include "machine_endian.h"

#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <assert.h>

int main (int argc, char** argv)
{
  fprintf (stderr, "\n\n"
                   "sizeof(BPP_SEARCH_HEADER) = %u\n"
                   "BPP_HEADER_SIZE = %u \n\n\n",
                   (unsigned) sizeof(BPP_SEARCH_HEADER), BPP_HEADER_SIZE);

  fprintf (stderr, "sizeof(long double) = %u \n\n\n", 
           (unsigned) sizeof(long double) );

  //assert (sizeof (BPP_SEARCH_HEADER) == BPP_HEADER_SIZE);

  if (argc == 1)
    return 0;

  FILE* fptr = fopen (argv[1], "r");
  if (!fptr)
  {
    fprintf (stderr, "Could not open '%s' %s \n", argv[1], strerror(errno));
    return -1;
  }

  BPP_SEARCH_HEADER bpp_search;

  fread(&bpp_search,sizeof(BPP_SEARCH_HEADER),1,fptr);

  FromBigEndian(bpp_search.header_version);

  fprintf (stderr, "header_version = %d\n", bpp_search.header_version);

  fclose (fptr);
  return 0;
}


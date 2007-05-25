/***************************************************************************
 *
 *   Copyright (C) 2007 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifdef HAVE_MALLOC_H
#include <malloc.h>
#endif

#include "dsp/Scratch.h"

// default scratch space used by Operations
dsp::Scratch dsp::Scratch::default_scratch;

dsp::Scratch::Scratch ()
{
  working_space = NULL;
  working_size = 0;
}

dsp::Scratch::~Scratch ()
{
  space (0);
}

//! Return pointer to a memory resource shared by operations
void* dsp::Scratch::space (size_t nbytes)
{

  if (!nbytes) {
    if (working_space) free(working_space); working_space = 0;
    working_size = 0;
  }

  if (working_size < nbytes) {
    if (working_space) free(working_space); working_space = 0;

#ifdef HAVE_MALLOC_H
    working_space = (char*)memalign(16, nbytes);
#else
    working_space = (char*)valloc(nbytes);
#endif

    if (!working_space)
      throw Error (BadAllocation, "Scratch::space",
	"error allocating %d bytes",nbytes);

    working_size = nbytes;
  }

  return working_space;
}


/***************************************************************************
 *
 *   Copyright (C) 2007 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Scratch.h"
#include "dsp/Memory.h"

#include <stdlib.h>

using namespace std;

// default scratch space used by Operations
static dsp::Scratch* the_default_scratch = 0;

dsp::Scratch* dsp::Scratch::get_default_scratch()
{
  if (!the_default_scratch)
    the_default_scratch = new Scratch;
  return the_default_scratch;
}

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
#ifdef _DEBUG
  cerr << "dsp::Scratch::space nbytes=" << nbytes 
       << " current=" << working_size << endl;
#endif

  if (!nbytes || working_size < nbytes)
  {
    if (working_space) Memory::free (working_space); working_space = 0;
  }

  if (!nbytes)
    return 0;

  if (working_space == 0)
  {
    working_space = (char*) Memory::allocate (nbytes);

    if (!working_space)
      throw Error (BadAllocation, "Scratch::space",
	"error allocating %d bytes",nbytes);

    working_size = nbytes;
  }

#ifdef _DEBUG
  cerr << "dsp::Scratch::space return start=" << (void*)working_space 
       << " end=" << (void*) (working_space + working_size) << endl;
#endif

  return working_space;
}


/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Memory.h"
#include "malloc16.h"

void* dsp::Memory::allocate (unsigned nbytes)
{
  // cerr << "dsp::Memory::allocate malloc16 (" << nbytes << ")" << endl;
  return malloc16 (nbytes);
}

void dsp::Memory::free (void* ptr)
{
  // cerr << "dsp::Memory::free free16 (" << ptr << ")" << endl;
  free16 (ptr);
}

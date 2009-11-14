/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Memory.h"
#include "malloc16.h"
#include <assert.h>

dsp::Memory* dsp::Memory::manager = new dsp::Memory;

void* dsp::Memory::do_allocate (unsigned nbytes)
{
  // cerr << "dsp::Memory::allocate malloc16 (" << nbytes << ")" << endl;
  return malloc16 (nbytes);
}

void dsp::Memory::do_free (void* ptr)
{
  // cerr << "dsp::Memory::free free16 (" << ptr << ")" << endl;
  free16 (ptr);
}

void* dsp::Memory::allocate (unsigned nbytes)
{
  return manager->do_allocate (nbytes);
}

void dsp::Memory::free (void* ptr)
{
  manager->do_free (ptr);
}

void dsp::Memory::set_manager (Memory* new_do)
{
  assert (new_do != 0);
  delete manager;
  manager = new_do;
}


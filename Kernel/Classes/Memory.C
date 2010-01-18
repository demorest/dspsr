/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Memory.h"
#include "malloc16.h"
#include "debug.h"

#include <assert.h>

dsp::Memory* dsp::Memory::manager = 0;

void* dsp::Memory::do_allocate (unsigned nbytes)
{
  DEBUG("dsp::Memory::allocate malloc16 (" << nbytes << ")");
  return malloc16 (nbytes);
}

void dsp::Memory::do_free (void* ptr)
{
  DEBUG("dsp::Memory::free free16 (" << ptr << ")");
  free16 (ptr);
}

void* dsp::Memory::allocate (unsigned nbytes)
{
  return get_manager()->do_allocate (nbytes);
}

void dsp::Memory::free (void* ptr)
{
  get_manager()->do_free (ptr);
}

// keep the manager alive when other Reference::To are used
static Reference::To<dsp::Memory> keep;

void dsp::Memory::set_manager (Memory* new_do)
{
  assert (new_do != 0);
  keep = manager = new_do;
}

dsp::Memory* dsp::Memory::get_manager ()
{
  if (!manager)
    keep = manager = new dsp::Memory;

  return manager;
}


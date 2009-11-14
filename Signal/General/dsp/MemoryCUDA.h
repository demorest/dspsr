//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_MemoryCUDA_h_
#define __dsp_MemoryCUDA_h_

#include "dsp/Memory.h"

namespace CUDA
{
  //! Manages CUDA pinned memory allocation and destruction
  class Memory : public dsp::Memory
  {
  public:
    void* allocate (unsigned nbytes);
    void free (void*);
  };
}

#endif

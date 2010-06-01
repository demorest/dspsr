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

#include <cuda_runtime.h>

namespace CUDA
{
  //! Manages CUDA pinned memory allocation and destruction
  class PinnedMemory : public dsp::Memory
  {
  public:
    void* do_allocate (unsigned nbytes);
    void do_free (void*);
  };

  //! Manages CUDA device memory allocation and destruction
  class DeviceMemory : public dsp::Memory
  {
  public:
    DeviceMemory (cudaStream_t _stream = 0) { stream = _stream; }

    void* do_allocate (unsigned nbytes);
    void do_free (void*);
    void do_copy (void* to, const void* from, size_t bytes);
    void do_zero (void*, unsigned);
    bool on_host () const { return false; }

    cudaStream_t get_stream () { return stream; }

  protected:
    cudaStream_t stream;
  };

  class SharedPinnedMemory : public dsp::Memory
  {
  public:
    void * do_allocate (unsigned nbytes);
    void do_free (void*);
  };
}

#endif

/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/MemoryCUDA.h"
#include <cuda_runtime.h>

void* CUDA::Memory::allocate (unsigned nbytes)
{
  // cerr << "CUDA::Memory::allocate cudaMallocHost (" << nbytes << ")" << endl;
  void* ptr = 0;
  cudaMallocHost (&ptr, nbytes);
  return ptr;
}

void CUDA::Memory::free (void* ptr)
{
  // cerr << "CUDA::Memory::free cudaFreeHost (" << ptr << ")" << endl;
  cudaFreeHost (ptr);
}

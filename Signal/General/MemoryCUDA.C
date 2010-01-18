/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/MemoryCUDA.h"

#include <cuda_runtime.h>

#include <iostream>
using namespace std;

void* CUDA::PinnedMemory::do_allocate (unsigned nbytes)
{
  cerr << "CUDA::Memory::allocate cudaMallocHost (" << nbytes << ")" << endl;
  void* ptr = 0;
  cudaMallocHost (&ptr, nbytes);
  return ptr;
}

void CUDA::PinnedMemory::do_free (void* ptr)
{
  cerr << "CUDA::Memory::free cudaFreeHost (" << ptr << ")" << endl;
  cudaFreeHost (ptr);
}

void* CUDA::DeviceMemory::do_allocate (unsigned nbytes)
{
  cerr << "CUDA::Memory::allocate cudaMalloc (" << nbytes << ")" << endl;
  void* ptr = 0;
  cudaMalloc (&ptr, nbytes);
  return ptr;
}

void CUDA::DeviceMemory::do_free (void* ptr)
{
  cerr << "CUDA::Memory::free cudaFree (" << ptr << ")" << endl;
  cudaFree (ptr);
}

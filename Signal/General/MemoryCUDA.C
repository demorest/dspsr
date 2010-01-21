/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/MemoryCUDA.h"
#include "debug.h"

#include <cuda_runtime.h>

#include <iostream>
using namespace std;

/***************************************************************************
 *
 *   Pinned memory on host
 *
 ***************************************************************************/

void* CUDA::PinnedMemory::do_allocate (unsigned nbytes)
{
  DEBUG("CUDA::PinnedMemory::allocate cudaMallocHost (" << nbytes << ")");
  void* ptr = 0;
  cudaMallocHost (&ptr, nbytes);
  return ptr;
}

void CUDA::PinnedMemory::do_free (void* ptr)
{
  DEBUG("CUDA::PinnedMemory::free cudaFreeHost (" << ptr << ")");
  cudaFreeHost (ptr);
}

void CUDA::PinnedMemory::do_copy (void* to, const void* from, size_t bytes)
{
  DEBUG("CUDA::PinnedMemory::copy (" << to <<","<< from <<","<< bytes << ")");
  cudaMemcpy (to, from, bytes, cudaMemcpyHostToHost);
}

/***************************************************************************
 *
 *   Memory on device
 *
 ***************************************************************************/

void* CUDA::DeviceMemory::do_allocate (unsigned nbytes)
{
  DEBUG("CUDA::DeviceMemory::allocate cudaMalloc (" << nbytes << ")" << endl;
  void* ptr = 0;
  cudaMalloc (&ptr, nbytes);
  return ptr;
}

void CUDA::DeviceMemory::do_free (void* ptr)
{
  cerr << "CUDA::DeviceMemory::free cudaFree (" << ptr << ")" << endl;
  cudaFree (ptr);
}

void CUDA::DeviceMemory::do_copy (void* to, const void* from, size_t bytes)
{
  DEBUG("CUDA::PinnedMemory::copy (" << to <<","<< from <<","<< bytes << ")");
  cudaMemcpy (to, from, bytes, cudaMemcpyDeviceToDevice);
}

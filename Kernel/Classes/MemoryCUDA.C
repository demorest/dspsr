/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

//#define _DEBUG

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

  cudaError error = cudaMallocHost (&ptr, nbytes);
  if (error != cudaSuccess)
    throw Error (FailedCall, "CUDA::PinnedMemory::do_allocate",
                 "cudaMallocHost (%x, %u): %s", &ptr, nbytes,
                 cudaGetErrorString (error));

  // error = cudaMemset (ptr, 0, nbytes);
  if (error != cudaSuccess)
    throw Error (FailedCall, "CUDA::PinnedMemory::do_allocate",
                 "cudaMemset (%x, 0, %u): %s", ptr, nbytes,
                 cudaGetErrorString (error));

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
  DEBUG("CUDA::DeviceMemory::allocate cudaMalloc (" << nbytes << ")");
  void* ptr = 0;
  cudaError err = cudaMalloc (&ptr, nbytes);
  if (err != cudaSuccess)
  {
    int device;
    cudaGetDevice (&device);
    throw Error (InvalidState, "CUDA::DeviceMemory::do_allocate",
                 "cudaMalloc failed on device %d: %s", device, cudaGetErrorString(err));
  }

  DEBUG("CUDA::DeviceMemory::allocate cudaMalloc ptr=" << ptr);
  return ptr;
}

void CUDA::DeviceMemory::do_free (void* ptr)
{
  DEBUG("CUDA::DeviceMemory::free cudaFree (" << ptr << ")");
  cudaFree (ptr);
}

void CUDA::DeviceMemory::do_copy (void* to, const void* from, size_t bytes)
{
  DEBUG("CUDA::DeviceMemory::copy (" << to <<","<< from <<","<< bytes << ")");
  cudaError err = cudaMemcpy (to, from, bytes, cudaMemcpyDeviceToDevice);
  if (err != cudaSuccess)
  {
    int device;
    cudaGetDevice (&device);
    throw Error (InvalidState, "CUDA::DeviceMemory::do_copy",
                 "cudaMemcpy failed on device %d: %s", device, cudaGetErrorString(err));
  }
}


/***************************************************************************
 *
 *   Shared pinned memory on host
 *
 ***************************************************************************/



void* CUDA::SharedPinnedMemory::do_allocate (unsigned nbytes)
{
  DEBUG("CUDA::SharedPinnedMemory::allocate cudaMallocHost (" << nbytes << ")");
  void* ptr = 0;
  cudaHostAlloc (&ptr, nbytes,cudaHostAllocPortable);
  return ptr;
}

void CUDA::SharedPinnedMemory::do_free (void* ptr)
{
  DEBUG("CUDA::SharedPinnedMemory::free cudaFreeHost (" << ptr << ")");
  cudaFreeHost (ptr);
}

void CUDA::SharedPinnedMemory::do_copy (void* to, const void* from, size_t bytes)
{
  DEBUG("CUDA::SharedPinnedMemory::copy (" << to <<","<< from <<","<< bytes << ")");
  cudaMemcpy (to, from, bytes, cudaMemcpyHostToHost);
}

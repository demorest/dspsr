//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2010 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SKMaskerCUDA.h"

#include <iostream>

using namespace std;

void check_error (const char*);

CUDA::SKMaskerEngine::SKMaskerEngine (cudaStream_t _stream)
{
  stream = _stream;
}

void CUDA::SKMaskerEngine::setup (unsigned _nchan, unsigned _npol, unsigned _span)
{
  if (dsp::Operation::verbose)
    cerr << "CUDA::SKMaskerEngine::setup nchan=" << _nchan << " npol=" << _npol
         << " span=" << _span << endl;

  nchan = _nchan;
  npol = _npol;
  span = _span;

  // determine GPU capabilities 
  int device = 0;
  cudaGetDevice(&device);
  struct cudaDeviceProp device_properties;
  cudaGetDeviceProperties (&device_properties, device);
  max_threads_per_block = device_properties.maxThreadsPerBlock;
}


/* cuda kernel to mask 1 channel for both polarisations */
__global__ void mask1chan (unsigned char * mask_base,
           float * out_base,
           unsigned npol,
           unsigned end,
           unsigned span)
{
  // ichan = blockIdx.x * blockDim.x + threadIdx.x

  float * p0 = out_base + span * npol * (blockIdx.x * blockDim.x + threadIdx.x);
  float * p1 = out_base + span * npol * (blockIdx.x * blockDim.x + threadIdx.x) + span;

  mask_base += (blockIdx.x * blockDim.x + threadIdx.x);

  if (mask_base[0])
  {
    for (unsigned j=0; j<end; j++)
    {
      p0[j] = 0;
      p1[j] = 0;
    }
  }

}

/*
 *  masks just 1 sample in the DDFB for the given SKFB channel and sample, uses __sync_threads
 *  with a __shared__ mask to improve read performance on access to the common mask value for all
 *  threads in a block
 */
__global__ void mask1sample (unsigned char * mask_base,
           float * out_base,
           unsigned npol,
           unsigned end,
           unsigned span)
{
  int ichan = blockIdx.x;

  __shared__ char mask;

  if (threadIdx.x == 0)
    mask = mask_base[ichan];

  __syncthreads();

  // zap if mask 
  if (mask)
  {
    int idat = threadIdx.x;
    int out_offset = (span * npol * ichan) + idat;

    out_base[out_offset] = 0;         // p0
    out_base[out_offset +span ] = 0;  // p1
  }
}


void CUDA::SKMaskerEngine::perform (dsp::BitSeries* mask, unsigned mask_offset, 
           dsp::TimeSeries * output, unsigned offset, unsigned end)
{

  if (dsp::Operation::verbose)
    cerr << "CUDA::SKMaskerEngine::perform mask_offset=" << mask_offset << " offset=" << offset << " end=" << end << endl;

  // order is FPT
  float * out_base = output->get_datptr(0, 0) + offset;
  unsigned char * mask_base = mask->get_datptr() + mask_offset;

  if (end > max_threads_per_block)
  {
    dim3 threads (128);
    dim3 blocks (nchan/threads.x);
    mask1chan<<<blocks,threads,0,stream>>> (mask_base, out_base, npol, end, span);
  }
  else
  {
    dim3 threads (end);
    dim3 blocks (nchan);
    mask1sample<<<blocks,threads,0,stream>>> (mask_base, out_base, npol, end, span);
  }

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error( "CUDA::SKMaskerEngine::perform" );
}


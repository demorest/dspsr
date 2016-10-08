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

CUDA::SKMaskerEngine::SKMaskerEngine (dsp::Memory * memory)
{
  device_memory = dynamic_cast<CUDA::DeviceMemory *>(memory);
  stream = device_memory->get_stream();
}

void CUDA::SKMaskerEngine::setup ()
{
  // determine GPU capabilities 
  int device = 0;
  cudaGetDevice(&device);
  struct cudaDeviceProp device_properties;
  cudaGetDeviceProperties (&device_properties, device);
  max_threads_per_block = device_properties.maxThreadsPerBlock;
}

/*
 *  masks just 1 sample in the DDFB for the given SKFB channel and sample, uses __sync_threads
 *  with a __shared__ mask to improve read performance on access to the common mask value for all
 *  threads in a block
 */
__global__ void mask1sample (unsigned char * mask_base,
           const float2 * in_base,
           float2 * out_base,
           uint64_t in_stride,
           uint64_t out_stride,
           uint64_t ndat,
           unsigned npol,
           unsigned M)
{
  const unsigned idat = blockIdx.x * blockDim.x + threadIdx.x; 
  if (idat >= ndat)
    return;

  const unsigned ichan = blockIdx.y;
  const unsigned imask = idat / M;

  // load the mask
  const unsigned char mask = mask_base[imask * gridDim.y + ichan];

  // forward pointer to pol0 for this chan
  out_base += ichan * npol * out_stride;
  in_base  += ichan * npol * in_stride;


  for (unsigned ipol=0; ipol<npol; ipol++)
  {
    if (mask)
    {
      out_base[idat].x = 0;
      out_base[idat].y = 0;
    }
    else
    {
      out_base[idat] = in_base[idat];
    }
    in_base  += in_stride;
    out_base += out_stride;
  }
}


void CUDA::SKMaskerEngine::perform (dsp::BitSeries* mask, const dsp::TimeSeries * input,
           dsp::TimeSeries * output, unsigned M)
{

  if (dsp::Operation::verbose)
    cerr << "CUDA::SKMaskerEngine::perform M=" << M << endl;
  
  // use output, since input may be InputBuffered
  uint64_t ndat  = output->get_ndat();
  unsigned nchan = output->get_nchan();
  unsigned npol  = output->get_npol();
  unsigned ndim  = output->get_ndim();
  // TODO assert that ndim == 2

  // order is FPT
  const float2 * in_base = (const float2 *) input->get_datptr (0, 0);
  float2 * out_base = (float2 *) output->get_datptr (0, 0);

  // order is TFP
  unsigned char * mask_base = mask->get_datptr();

  uint64_t in_stride, out_stride;
  if (npol == 1)
  {
    in_stride = input->get_datptr (1, 0) - input->get_datptr (0, 0);
    out_stride = output->get_datptr (1, 0) - output->get_datptr (0, 0);
  }
  else
  {
    in_stride = input->get_datptr (0, 1) - input->get_datptr (0, 0);
    out_stride = output->get_datptr (0, 1) - output->get_datptr (0, 0);
  }

  // strides are numbers of floats between
  in_stride /= ndim;
  out_stride /= ndim;

  unsigned threads = max_threads_per_block;
  dim3 blocks (ndat / threads, nchan);
  if (ndat % threads)
    blocks.x++;

#ifdef _DEBUG
  cerr << "CUDA::SKMaskerEngine::perform ndat=" << ndat << " nchan=" << nchan << " npol=" << npol << " ndim=" << ndim << endl;
  cerr << "CUDA::SKMaskerEngine::perform in_stride=" << in_stride << " out_stride=" << out_stride << endl;
  cerr << "CUDA::SKMaskerEngine::perform blocks=(" << blocks.x << ", " << blocks.y << ") threads=" << threads << endl;
#endif

  mask1sample<<<blocks,threads,0,stream>>> (mask_base, in_base, out_base, in_stride, out_stride, ndat, npol, M);

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error( "CUDA::SKMaskerEngine::perform" );
}


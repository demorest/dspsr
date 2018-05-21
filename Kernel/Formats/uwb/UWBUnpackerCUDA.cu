//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <stdio.h>
#include <cuda_runtime.h>

#include "dsp/UWBUnpackerCUDA.h"
#include "dsp/Operation.h"
#include "dsp/MemoryCUDA.h"

#include "Error.h"
#define WARP_SIZE 32
#define NSAMP_PER_BLOCK 2048
#define NPOL 2

using namespace std;

void check_error_stream (const char*, cudaStream_t);

__inline__ __device__ int16_t convert_offset_binary(int16_t in) { return  in^0x8000; };

/* 1024 threads, and 2048 samples per block */
__global__ void uwb_unpack_fpt_2pol_kernel (float2 * to_a, float2 * to_b, 
                                       const int32_t * from, uint64_t ndat)
{
  const unsigned in_pol_span = 2048;

  // input indexing has 2 polarisations interleaved
  uint64_t idat = (blockIdx.x * NSAMP_PER_BLOCK * NPOL) + threadIdx.x;
  // output indexing has 2 polarisations separate
  uint64_t odat = (blockIdx.x * NSAMP_PER_BLOCK) + threadIdx.x;

  if (odat >= ndat)
    return;

  int32_t packed32;
  int16_t * packed16 = (int16_t *) &packed32;
  float2 unpacked;

  // unpack pol 0, samp 0+idx
  packed32 = from[idat];
  unpacked.x = float(convert_offset_binary(packed16[0]));
  unpacked.y = float(convert_offset_binary(packed16[1]));
  to_a[odat] = unpacked;

  // unpack pol 1 samp 0+idx
  packed32 = from[idat + in_pol_span];
  unpacked.x = float(convert_offset_binary(packed16[0]));
  unpacked.y = float(convert_offset_binary(packed16[1]));
  to_b[odat] = unpacked;

  idat += 1024;
  odat += 1024;

  if (odat >= ndat)
    return;

  // unpack pol 0 samp 1024+idx
  packed32 = from[idat];
  unpacked.x = float(convert_offset_binary(packed16[0]));
  unpacked.y = float(convert_offset_binary(packed16[1]));
  to_a[odat] = unpacked;

  // unpack pol 1 samp 1024+idx
  packed32 = from[idat + in_pol_span];
  unpacked.x = float(convert_offset_binary(packed16[0]));
  unpacked.y = float(convert_offset_binary(packed16[1]));
  to_b[odat] = unpacked;
}

/* 1024 threads, and 2048 samples per block */
__global__ void uwb_unpack_fpt_1pol_kernel (float2 * to, const int32_t * from, uint64_t ndat, bool first_block)
{
  uint64_t idx = (blockIdx.x * NSAMP_PER_BLOCK) + threadIdx.x;

  if (idx >= ndat)
    return;

  int32_t packed32;
  int16_t * packed16 = (int16_t *) &packed32;
  float2 unpacked;

  // unpack pol 0, samp 0+idx
  packed32 = from[idx];
  unpacked.x = float(convert_offset_binary(packed16[0]));
  unpacked.y = float(convert_offset_binary(packed16[1]));
  to[idx] = unpacked;

  idx += 1024;

  if (idx >= ndat)
    return;

  // unpack pol 0 samp 1024+idx
  packed32 = from[idx];
  unpacked.x = float(convert_offset_binary(packed16[0]));
  unpacked.y = float(convert_offset_binary(packed16[1]));
  to[idx] = unpacked;
}


CUDA::UWBUnpackerEngine::UWBUnpackerEngine (cudaStream_t _stream)
{
  stream = _stream;
  first_block = true;
}

void CUDA::UWBUnpackerEngine::setup ()
{
  // determine cuda device properties for block & grid size
  int device;
  cudaGetDevice(&device);
  cudaGetDeviceProperties (&gpu, device);
}

bool CUDA::UWBUnpackerEngine::get_device_supported (dsp::Memory* memory) const
{
  return dynamic_cast< CUDA::DeviceMemory*> ( memory );
}

void CUDA::UWBUnpackerEngine::set_device (dsp::Memory* memory)
{
}

void CUDA::UWBUnpackerEngine::unpack (const dsp::BitSeries * input, dsp::TimeSeries * output)
{
  // data are packed in 2048 sample heaps, each block unpacks a heap of 2048
  // samples from both polarisations

  const uint64_t ndat = input->get_ndat();
  const int npol = input->get_npol();

  unsigned nthreads = 1024;
  unsigned nblocks = ndat / NSAMP_PER_BLOCK;
  if (ndat % NSAMP_PER_BLOCK > 0)
    nblocks++;

  // use an int32_t to handle the re+im parts of the int16_t
  int32_t * from = (int32_t *) input->get_rawptr();

  if (npol == 2)
  {
    float2  * into_a = (float2 *) output->get_datptr(0, 0);
    float2  * into_b = (float2 *) output->get_datptr(0, 1);
    uwb_unpack_fpt_2pol_kernel<<<nblocks,nthreads,0,stream>>> (into_a, into_b, from, ndat);
  }
  else
  {
    float2  * into = (float2 *) output->get_datptr(0, 0);
    uwb_unpack_fpt_1pol_kernel<<<nblocks,nthreads,0,stream>>> (into, from, ndat, first_block);
  }

  first_block = false;
  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error_stream ("CUDA::UWBUnpackerEngine::unpack", stream);
}

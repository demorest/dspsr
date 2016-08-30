//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2015 by Matthew Kerr
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/TScrunchCUDA.h"

#include <cuComplex.h>
#include "Error.h"
#include "debug.h"

//#include <memory>

//#include <string.h>

using namespace std;

void check_error (const char*);

CUDA::TScrunchEngine::TScrunchEngine (cudaStream_t _stream)
{
  stream = _stream;
}

__global__ void fpt_ndim2_ndim2 (float2* in_base, float2* out_base,
    unsigned in_Fstride, unsigned in_Pstride, 
    unsigned out_Fstride, unsigned out_Pstride,
    unsigned output_ndat, unsigned sfactor)
{

  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= output_ndat)
    return;

  // blockIdx.y == channel index
  // threadIdx.y == polarization index
  // offset into buffer = the index of the output datum (i) * the scrunch factor
  in_base += blockIdx.y * in_Fstride + threadIdx.y * in_Pstride + sfactor * i;
  float2 result = *in_base;
  for (int j=1; j < sfactor; ++j,++in_base)
  {
    result.x += (*in_base).x;
    result.y += (*in_base).y;
  }

  out_base += blockIdx.y * out_Fstride +threadIdx.y * out_Pstride + i;
  *out_base = result;
}

__global__ void fpt_ndim2_ndim2_shm (float2* in_base, float2* out_base,
    unsigned in_Fstride, unsigned in_Pstride,
    unsigned out_Fstride, unsigned out_Pstride,
    unsigned ndat_out, unsigned sfactor)
{
  // shared memory for coalesced reads
  extern __shared__ cuFloatComplex shm[];

  // blockIdx.y == channel index
  // threadIdx.y == polarization index
  unsigned ndat_in = ndat_out * sfactor;

  const unsigned block_offset = blockIdx.x * blockDim.x * sfactor;

  // X dimension is indexed on output samples. This is the input sample each thread will start to read
  unsigned isamp_thr = block_offset + threadIdx.x;
 
  // offset into buffer = the index the first read sample for this block
  in_base += (blockIdx.y*in_Fstride) + (threadIdx.y*in_Pstride) + block_offset;

  cuFloatComplex result = make_cuComplex(0,0);
  unsigned isamp = threadIdx.x * sfactor;
  unsigned esamp = isamp + sfactor;
  unsigned shm_start = 0;
  unsigned shm_end = blockDim.x;

  // ensure we don't overshoot the number of ndat
  for (unsigned j=0; j<sfactor; j++)
  {
    // just whole block to coalesce read into SHM
    if (isamp_thr < ndat_in)
      shm[threadIdx.x] = in_base[isamp_thr];

    __syncthreads();

    // each thread adds time samples into its output result, wait for
    // the right time samples to be located in shm

    // if this thread's output value is located in SHM, add to result
    while (isamp >= shm_start && isamp < shm_end && isamp < esamp)
    {
      //if (blockIdx.y == 0 && blockIdx.z == 0) 
      //  printf ("[%d][%d] isamp=%u esamp=%u start=%u end=%u\n", blockIdx.x, threadIdx.x, isamp, esamp, shm_start, shm_end);
      result = cuCaddf (result, shm[isamp-shm_start]);
      isamp++;
    }

    isamp_thr += blockDim.x;
    shm_start += blockDim.x;
    shm_end   += blockDim.x;
  }
 
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ndat_out)
    return;

  //if (blockIdx.y == 0 && blockIdx.z == 0)
  //  printf ("[%d][%d] i=%u\n", blockIdx.x, threadIdx.x, i);

  out_base += (blockIdx.y*out_Fstride) + (threadIdx.y*out_Pstride) + i;
  *out_base = result;
}


void CUDA::TScrunchEngine::fpt_tscrunch(const dsp::TimeSeries *in,
    dsp::TimeSeries* out, unsigned sfactor)
{
  // the "inner loop", which each thread does, is the tscrunch itself
  
  // the theory is that if one is time scrunching on the GPU, the 
  // scrunch factor will be something of order 10-100, a reasonable
  // amount of work for a thread to do

  // this is not at all optimal in terms of cache access, and at some point
  // this should be re-written with each thread accessing adjacent samples

  // to manage restrictions on grid size in earlier compute capability,
  // use a 2d thread block with one dimension corresponding to P,
  // the other to the output T
  // then launch on a 2d grid with one block handling the full output size, the
  // other handling the channels

  // each thread is assigned to a specific input F & input P
  // and loops over input T to add sfactor data together 

  // currently the only implementation uses float2s so we require
  // ndim==2 for both input and output
  if (in->get_ndim() != 2)
    throw Error (InvalidParam, "CUDA::TScrunchEngine::fpt_scrunch",
		 "cannot handle ndim=%u != 2", in->get_ndim());

  if (out->get_ndim() != 2)
    throw Error (InvalidParam, "CUDA::TScrunchEngine::fpt_scrunch",
		 "cannot handle ndim=%u != 2", in->get_ndim());

  if (out == in)
    throw Error (InvalidParam, "CUDA::TScrunchEngine::fpt_scrunch",
		 "only out-of-place transformation implemented");

  if (in->get_ndat() == 0)
    return;

  uint64_t in_Fstride = (in->get_datptr(1)-in->get_datptr(0)) / 2;
  uint64_t in_Pstride = (in->get_datptr(0,1)-in->get_datptr(0,0)) / 2;
  uint64_t out_Fstride = (out->get_datptr(1)-out->get_datptr(0)) / 2;
  uint64_t out_Pstride = (out->get_datptr(0,1)-out->get_datptr(0,0)) / 2;
  // use a 2-dimensional thread block to eliminate 3rd grid dimension

#define USE_SHARED
#ifdef USE_SHARED
  // set number of threads to be number of output samples, cap at 512
  dim3 threads (512);
  if (out->get_ndat() < 512)
    threads.x = out->get_ndat();
  dim3 blocks (out->get_ndat()/threads.x, in->get_nchan(), in->get_npol());
  if (out->get_ndat() % threads.x)
    blocks.x ++;

  size_t shm_bytes = threads.x * sizeof(float2);
  fpt_ndim2_ndim2_shm<<<blocks,threads,shm_bytes,stream>>> (
    (float2*)(in->get_datptr(0)), (float2*)(out->get_datptr(0)), 
    in_Fstride, in_Pstride, out_Fstride, out_Pstride, 
    out->get_ndat(), sfactor);
#else
  dim3 threads (128, in->get_npol());
  dim3 blocks (out->get_ndat()/threads.x, in->get_nchan(), in->get_npol());
  if (out->get_ndat() % threads.x)
    blocks.x ++;
  fpt_ndim2_ndim2<<<blocks,threads,0,stream>>> (
    (float2*)(in->get_datptr(0)), (float2*)(out->get_datptr(0)), 
    in_Fstride, in_Pstride, out_Fstride, out_Pstride, 
    out->get_ndat(), sfactor);
#endif


  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error ("CUDA::TScrunchEngine::fpt_scrunch");
}


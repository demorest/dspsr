//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2015 by Matthew Kerr
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/TScrunchCUDA.h"

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
  // blockIdx.z == polarization index
  // offset into buffer = the index of the output datum (i) * the scrunch factor
  in_base += blockIdx.y * in_Fstride + blockIdx.z * in_Pstride + sfactor * i;
  float2 result = *in_base;
  for (int j=1; j < sfactor; ++j,++in_base)
  {
    result.x += (*in_base).x;
    result.y += (*in_base).y;
  }

  out_base += blockIdx.y * out_Fstride + blockIdx.z * out_Pstride + i;
  *out_base = result;
}

void CUDA::TScrunchEngine::fpt_tscrunch(const dsp::TimeSeries *in,
    dsp::TimeSeries* out, unsigned sfactor)
{
  // the "inner loop", which each thread does, is the tscrunch itself
  
  // the theory is that if one is time scrunching on the GPU, the 
  // scrunch factor will be something of order 10-100, a reasonable
  // amount of work for a thread to do

  // the kernel is launched on a 3d block, with the dimensions
  // corresponding to output T, input/output F, input/output P
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

  uint64_t in_Fstride = (in->get_datptr(1)-in->get_datptr(0)) / 2;
  uint64_t in_Pstride = (in->get_datptr(0,1)-in->get_datptr(0,0)) / 2;
  uint64_t out_Fstride = (out->get_datptr(1)-out->get_datptr(0)) / 2;
  uint64_t out_Pstride = (out->get_datptr(0,1)-out->get_datptr(0,0)) / 2;
  dim3 threads (128);
  dim3 blocks (out->get_ndat()/threads.x, in->get_nchan(), in->get_npol());

  if (out->get_ndat() % threads.x)
    blocks.x ++;

  fpt_ndim2_ndim2<<<blocks,threads,0,stream>>> (
    (float2*)(in->get_datptr(0)), (float2*)(out->get_datptr(0)), 
    in_Fstride, in_Pstride, out_Fstride, out_Pstride, 
    out->get_ndat(), sfactor);

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error ("CUDA::TScrunchEngine::fpt_scrunch");
}


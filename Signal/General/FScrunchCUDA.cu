//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2015 by Matthew Kerr
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/FScrunchCUDA.h"

#include "Error.h"
#include "debug.h"

//#include <memory>

//#include <string.h>

using namespace std;

void check_error (const char*);

CUDA::FScrunchEngine::FScrunchEngine (cudaStream_t _stream)
{
  stream = _stream;
}

__global__ void fpt_ndim2_ndim2 (float2* in_base, float2* out_base,
    unsigned in_span, unsigned out_span, unsigned ndat, unsigned sfactor)
{

  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ndat)
    return;

  in_base += blockIdx.y * in_span * sfactor + i;
  float2 result = *in_base;
  float2 tmp;
  for (int j=1; j < sfactor; ++j)
  {
    tmp = *(in_base + j*in_span);
    result.x += tmp.x;
    result.y += tmp.y;
  }

  out_base += blockIdx.y * out_span + i;
  *out_base = result;
}

void CUDA::FScrunchEngine::fpt_fscrunch(const dsp::TimeSeries *in,
    dsp::TimeSeries* out, unsigned sfactor)
{
  // set up two-dimensional blocks; the first dimension corresponds to an
  // index along the data rows (so polarization, time, and dimension), and
  // the second to output channel; each thread will add up the rows for
  // sfactor input channels and write out to a single output channel; this
  // avoid any need for synchronization

  // initial implementation requires input and output in dimension 2
  // and uses float2 for speed
  if (in->get_ndim() != 2)
    throw Error (InvalidParam, "CUDA::FScrunchEngine::fpt_scrunch",
		 "cannot handle ndim=%u != 2", in->get_ndim());

  if (out->get_ndim() != 2)
    throw Error (InvalidParam, "CUDA::FScrunchEngine::fpt_scrunch",
		 "cannot handle ndim=%u != 2", in->get_ndim());

  // TODO -- enforce out of place?  Technically could work with in place.

  // number of float2s between adjacent input frequency channels
  const float* in_base = in->get_datptr(0);
  uint64_t in_span = (in->get_datptr(1)-in_base)/2;

  // number of float2s between adjacent output frequency channels
  float* out_base = out->get_datptr(0);
  uint64_t out_span = (out->get_datptr(1)-out_base)/2;

  uint64_t ndat = in->get_ndat ();
  dim3 threads (128);
  dim3 blocks (ndat/threads.x, in->get_nchan()/sfactor);

  if (ndat % threads.x)
    blocks.x ++;

  fpt_ndim2_ndim2<<<blocks,threads,0,stream>>> (
    (float2*)(in_base), (float2*)(out_base), 
    in_span, out_span, ndat, sfactor);

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error ("CUDA::FScrunchEngine::fpt_scrunch");
}


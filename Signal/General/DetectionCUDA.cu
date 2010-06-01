//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/DetectionCUDA.h"

#include "Error.h"
#include "cross_detect.h"
#include "stokes_detect.h"
#include "templates.h"
#include "debug.h"

#include <memory>

#include <string.h>

using namespace std;

/*
  PP   = p^* p
  QQ   = q^* q
  RePQ = Re[p^* q]
  ImPQ = Im[p^* q]
*/

#define COHERENCE(PP,QQ,RePQ,ImPQ,p,q) \
  PP   = (p.x * p.x) + (p.y * p.y); \
  QQ   = (q.x * q.x) + (q.y * q.y); \
  RePQ = (p.x * q.x) + (p.y * q.y); \
  ImPQ = (p.x * q.y) - (p.y * q.x);


#define COHERENCE4(r,p,q) COHERENCE(r.w,r.x,r.y,r.z,p,q)

__global__ void coherence4 (float4* base, uint64_t span)
{
  base += blockIdx.y * span;
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

  float2 p, q;
  float4 result = base[i];

  p.x = result.w;
  p.y = result.x;
  q.x = result.y;
  q.y = result.z;

  COHERENCE4 (result,p,q);

  base[i] = result;
}

/*
  The input data are arrays of ndat pairs of complex numbers: 

  Re[p0],Im[p0],Re[p1],Im[p1]

  There are nchan such arrays; base pointers are separated by span.
*/

void polarimetry_ndim4 (float* data, uint64_t span,
			uint64_t ndat, unsigned nchan)
{
  int threads = 256;

  dim3 blocks;
  blocks.x = ndat/threads;
  blocks.y = nchan;

  coherence4<<<blocks,threads>>> ((float4*)data, span/4); 

  cudaThreadSynchronize();
  cudaError error = cudaGetLastError();
  if (error != cudaSuccess)
    cerr << "FAIL coherence: " << cudaGetErrorString (error) << endl;
}

/*
  The input data are pairs of arrays of ndat complex numbers: 

  Re[p0],Im[p0] ...

  Re[p1],Im[p1] ...

  There are nchan such pairs of arrays; base pointers p0 and p1 are 
  separated by span.
*/

#define COHERENCE2(s0,s1,p,q) COHERENCE(s0.x,s0.y,s1.x,s1.y,p,q)

__global__ void coherence2 (float2* base, unsigned span, unsigned ndat)
{
  float2* p0 = base + blockIdx.y * span * 2;
  float2* p1 = p0 + span;

  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= ndat)
    return;

  float2 s0, s1;

  COHERENCE2(s0,s1,p0[i],p1[i]);

  p0[i] = s0;
  p1[i] = s1;
}

void polarimetry_ndim2 (float* data, uint64_t span,
			uint64_t ndat, unsigned nchan)
{
  dim3 threads (128);
  dim3 blocks (ndat/threads.x, nchan);

  if (ndat % threads.x)
    blocks.x ++;

  DEBUG("polarimetry_ndim2 ndat=" << ndat << " span=" << span \
        << " blocks=" << blocks.x << " threads=" << threads.x \
        << " data=" << data);

  // pass span as number of complex values
  coherence2<<<blocks,threads>>> ((float2*)data, span/2, ndat); 

  cudaThreadSynchronize();
  cudaError error = cudaGetLastError();
  if (error != cudaSuccess)
    cerr << "FAIL coherence: " << cudaGetErrorString (error) << endl;
}

void CUDA::DetectionEngine::polarimetry (unsigned ndim,
					 const dsp::TimeSeries* input, 
					 dsp::TimeSeries* output)
{
  if (ndim != 2)
    throw Error (InvalidParam, "CUDA::DetectionEngine::polarimetry",
		 "cannot handle ndim=%u != 2", ndim);

  if (input != output)
    throw Error (InvalidParam, "CUDA::DetectionEngine::polarimetry"
		 "cannot handle out-of-place data");

  uint64_t ndat = output->get_ndat ();
  unsigned nchan = output->get_nchan ();

  unsigned ichan=0, ipol=0;

  float* base = output->get_datptr (ichan=0, ipol=0);

  uint64_t span = output->get_datptr (ichan=0, ipol=1) - base;

  if (dsp::Operation::verbose)
    cerr << "CUDA::DetectionEngine::polarimetry ndim=" << output->get_ndim () 
         << " ndat=" << ndat << " span=" << span << endl;

  polarimetry_ndim2 (base, span, ndat, nchan);

  if (dsp::Operation::record_time)
  {
    cudaThreadSynchronize ();
    cudaError error = cudaGetLastError();
    if (error != cudaSuccess)
      throw Error (InvalidState, "CUDA::DetectionEngine::polarimetry", 
                   cudaGetErrorString (error));
  }
}


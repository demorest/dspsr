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

void check_error (const char*);

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

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error ("CUDA::DetectionEngine::polarimetry_ndim4");
}

/*
  The input data are pairs of arrays of ndat complex numbers: 

  Re[p0],Im[p0] ...

  Re[p1],Im[p1] ...

  There are nchan such pairs of arrays; base pointers p0 and p1 are 
  separated by span.
*/

#define COHERENCE2(s0,s1,p,q) COHERENCE(s0.x,s0.y,s1.x,s1.y,p,q)

__global__ void coherence2 (const float2* input_base, unsigned input_span, 
                            float2* output_base, unsigned output_span,
                            unsigned ndat)
{
#define COHERENCE_NPOL 2
  const float2* p0 = input_base + blockIdx.y * input_span * COHERENCE_NPOL;
  const float2* p1 = p0 + input_span;

  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= ndat)
    return;

  float2 s0, s1;

  COHERENCE2(s0,s1,p0[i],p1[i]);

  float2* op0 = output_base + blockIdx.y * output_span * COHERENCE_NPOL;
  float2* op1 = op0 + output_span;

  op0[i] = s0;
  op1[i] = s1;
}


CUDA::DetectionEngine::DetectionEngine (cudaStream_t _stream)
{
  stream = _stream;
}

void CUDA::DetectionEngine::polarimetry (unsigned ndim,
					 const dsp::TimeSeries* input, 
					 dsp::TimeSeries* output)
{
  if (ndim != 2)
    throw Error (InvalidParam, "CUDA::DetectionEngine::polarimetry",
		 "cannot handle ndim=%u != 2", ndim);

  uint64_t ndat = input->get_ndat ();
  unsigned ichan_start = input->get_ichan_start();
  unsigned nchan_bundle = input->get_nchan_bundle();

  if (ndat != output->get_ndat ())
    throw Error (InvalidParam, "CUDA::DetectionEngine::polarimetry",
                 "input ndat=%u != output ndat=%u",
                 ndat, output->get_ndat());

  unsigned ichan=0, ipol=0;

  const float* input_base = input->get_datptr (ichan=ichan_start, ipol=0);
  uint64_t input_span = input->get_datptr (ichan=ichan_start, ipol=1) - input_base;

  float* output_base = output->get_datptr (ichan=ichan_start, ipol=0);
  uint64_t output_span = output->get_datptr (ichan=ichan_start, ipol=1) - output_base;

  if (dsp::Operation::verbose)
    cerr << "CUDA::DetectionEngine::polarimetry ndim=" << output->get_ndim () 
         << " ndat=" << ndat 
         << " input.base=" << input_base
         << " output.base=" << output_base
         << " input.span=" << input_span 
         << " output.span=" << output_span << endl;

  dim3 threads (128);
  dim3 blocks (ndat/threads.x, nchan_bundle);

  if (ndat % threads.x)
    blocks.x ++;

  // pass span as number of complex values
  coherence2<<<blocks,threads,0,stream>>> ((const float2*)input_base, 
                                           input_span/2,
                                           (float2*)output_base,
                                           output_span/2,
                                           ndat); 

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error ("CUDA::DetectionEngine::polarimetry");
}


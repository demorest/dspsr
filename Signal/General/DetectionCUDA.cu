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
void check_error_stream (const char*, cudaStream_t);

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
  unsigned nchan = input->get_nchan ();

  if (ndat != output->get_ndat ())
    throw Error (InvalidParam, "CUDA::DetectionEngine::polarimetry",
                 "input ndat=%u != output ndat=%u",
                 ndat, output->get_ndat());

  unsigned ichan=0, ipol=0;

  const float* input_base = input->get_datptr (ichan=0, ipol=0);
  uint64_t input_span = input->get_datptr (ichan=0, ipol=1) - input_base;

  float* output_base = output->get_datptr (ichan=0, ipol=0);
  uint64_t output_span = output->get_datptr (ichan=0, ipol=1) - output_base;

  if (dsp::Operation::verbose)
    cerr << "CUDA::DetectionEngine::polarimetry ndim=" << output->get_ndim () 
         << " ndat=" << ndat 
         << " input.base=" << input_base
         << " output.base=" << output_base
         << " input.span=" << input_span 
         << " output.span=" << output_span << endl;

  dim3 threads (128);
  dim3 blocks (ndat/threads.x, nchan);

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

// dubiuous about the correctness here... TODO AJ
__global__ void sqld_tfp (float2 *base_in, unsigned stride_in,
                          float * base_out, unsigned stride_out, unsigned ndat)
{
  // input and output pointers for channel (y dim)
  float2 * in = base_in + (blockIdx.y * stride_in);
  float * out = base_out + (blockIdx.y * stride_out);

  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

  out[i] = in[i].x * in[i].x + in[i].y * in[i].y;
}

__global__ void sqld_fpt (float2 *base_in, float *base_out, uint64_t ndat)
{
  // set base pointer for ichan [blockIdx.y], input complex, output detected, npol 1
  float2 * in = base_in + (blockIdx.y * ndat);
  float * out = base_out + (blockIdx.y * ndat);

  // the sample for the channel
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

  out[i] = in[i].x * in[i].x + in[i].y * in[i].y;
}

void CUDA::DetectionEngine::square_law (const dsp::TimeSeries* input,
           dsp::TimeSeries* output)
{
  uint64_t ndat  = input->get_ndat ();
  unsigned nchan = input->get_nchan ();
  unsigned ndim  = input->get_ndim();
  unsigned npol  = input->get_npol();

  if (ndim != 2)
    throw Error (InvalidParam, "CUDA::DetectionEngine::square_law",
     "cannot handle ndim=%u != 2", ndim);

  if (npol != 1)
    throw Error (InvalidParam, "CUDA::DetectionEngine::square_law",
     "cannot handle npol=%u != 1", ndim);

  if (input == output)
    throw Error (InvalidParam, "CUDA::DetectionEngine::square_law"
     "cannot handle in-place data");

/*
  if (input->get_order() == dsp::TimeSeries::OrderTFP)
    cerr << "CUDA::DetectionEngine::square_law input->get_order=TFP" << endl;
  if (output->get_order() == dsp::TimeSeries::OrderTFP)
    cerr << "CUDA::DetectionEngine::square_law output->get_order=TFP" << endl;
*/

  switch (input->get_order())
  {
    case dsp::TimeSeries::OrderTFP:
    {
      dim3 threads (512);
      dim3 blocks (ndat/threads.x, nchan);

      if (ndat % threads.x)
        blocks.x ++;

      float2* base_in = (float2*) input->get_dattfp ();
      float* base_out = output->get_dattfp();

      unsigned stride_in = nchan * npol;
      unsigned stride_out = nchan * npol;

      if (dsp::Operation::verbose)
        cerr << "CUDA::DetectionEngine::square_law sqld_tfp ndat=" << ndat
             << " stride_in=" << stride_in << " stride_out=" << stride_out << endl;

      sqld_tfp<<<blocks,threads,0,stream>>> (base_in, stride_in, base_out, stride_out, ndat);

      if (dsp::Operation::record_time || dsp::Operation::verbose)
        check_error_stream ("CUDA::DetectionEngine::square_law sqld_tfp", stream);

      break;
    }

    case dsp::TimeSeries::OrderFPT:
    {
      dim3 threads (512);
      dim3 blocks (ndat/threads.x, nchan);

      if (ndat % threads.x)
        blocks.x ++;

      unsigned ichan = 0;
      unsigned ipol = 0;
      float2* base_in = (float2*) input->get_datptr(ichan, ipol);
      float* base_out = output->get_datptr(ichan, ipol);

      if (dsp::Operation::verbose)
        cerr << "CUDA::DetectionEngine::square_law <<<sqld_fpt>>> "
             << " base_in=" << (void *) base_in
             << " base_out=" << (void *) base_out
             << " ndat=" << ndat << endl;

      sqld_fpt<<<blocks,threads,0,stream>>> (base_in, base_out, ndat);

      if (dsp::Operation::record_time || dsp::Operation::verbose)
        check_error_stream ("CUDA::DetectionEngine::square_law sqld_fpt", stream);

      break;
    }

    default:
    {
      throw Error (InvalidState, "CUDA::DetectionEngine::square_law", "unrecognized order");
    }
  }
}

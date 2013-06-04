//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2013 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/MOPSRUnpackerCUDA.h"
#include "dsp/Operation.h"
#include "debug.h"
#include "Error.h"

// threads per block - C1060=256 [TODO CHECK below if changing]
#define __MOPSR_UNPACK_TPB 512

#define __MOPSR_SAMPLES_PER_THREAD 4

// global static texture declaration for MOPSR gpu unpacker
//texture<char2, cudaTextureType1D, cudaReadModeNormalizedFloat> mopsr_tex1dfloat2;

// real textutre version
//texture<int8_t, cudaTextureType1D, cudaReadModeElementType> mopsr_tex1dfloat;

using namespace std;

void check_error (const char*);
void check_error_stream (const char*, cudaStream_t);

__device__ __constant__ float mopsr_unpacker_scale;

#ifdef USE_TEXTURE_MEMORY
__global__ void mopsr_unpack_complex_1 (float2 * output, cudaTextureObject_t tex)
{
  const int idx = blockIdx.x*blockDim.x + threadIdx.x;
  output[idx] = tex1Dfetch<float2>(tex, idx);
}
#else
__global__ void mopsr_unpack_fpt_complex_1 (const int8_t * stagingBufGPU,
                                            float2 * output, 
                                            const unsigned chan_stride,
                                            const unsigned samp_stride)
{
  const unsigned isamp = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned ichan = blockIdx.y;

  // input will be in TFP order
  // output will be in FTP order

  const int8_t* from = reinterpret_cast<const int8_t*>( stagingBufGPU ) + (isamp * samp_stride) + (ichan * 2);
  //float2 * to        = output + (ichan * chan_stride) + isamp * 2;

  __shared__ float2 out;

  out.x = (float) from[0];
  out.y = (float) from[1];

  // + 0.5 since scale is -128 to 127
  out.x += 0.5;
  out.y += 0.5;

  // optimal scaling from bit table
  out.x *= mopsr_unpacker_scale;
  out.y *= mopsr_unpacker_scale;

  output[(ichan * chan_stride) + (isamp)] = out;
}
#endif

__global__ void mopsr_unpack_tfp_complex_1 (const int8_t * stagingBufGPU,
                                            float2* output,
                                            const unsigned nchan)
{
  const unsigned isamp = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned ichan = blockIdx.y;
  const unsigned ndim = 2;

  // input and output will be in TFP order
  const int8_t* from = reinterpret_cast<const int8_t*>( stagingBufGPU ) + (isamp * nchan * ndim) + (ichan * ndim);

  __shared__ float2 out;

  out.x = (float) from[0];
  out.y = (float) from[1];

  // + 0.5 since scale is -128 to 127
  out.x += 0.5;
  out.y += 0.5;

  // optimal scaling from bit table
  out.x *= mopsr_unpacker_scale;
  out.y *= mopsr_unpacker_scale;

  output[(isamp * nchan) + ichan] = out;
}


void mopsr_unpack_prepare (const float scale)
{
  cerr << "mopsr_unpack_prepare: setting scale to " << scale << endl;
  cudaMemcpyToSymbol(mopsr_unpacker_scale, &scale, sizeof(scale));
}

void mopsr_unpack_tfp (cudaStream_t stream, const uint64_t ndat, const unsigned nchan,
                          float scale, int8_t const * input, float * output)
{
  int nthread = __MOPSR_UNPACK_TPB;
  int nblocks = ndat / nthread;

  // each thread will unpack 1 complex time sample from 1 channel
  dim3 blocks (nblocks, nchan);

  float2 * complex_output = (float2 *) output;

  mopsr_unpack_tfp_complex_1<<<blocks,nthread,0,stream>>>(input, complex_output, nchan);

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error_stream ("mopsr_unpack_tfp", stream);
}

#ifdef USE_TEXTURE_MEMORY
void mopsr_unpack (cudaStream_t stream, const uint64_t ndat,
                   unsigned char const* input, float * output,
                   cudaTextureObject_t * tex)
#else
void mopsr_unpack_fpt (cudaStream_t stream, const uint64_t ndat, const unsigned nchan,
                       float scale, int8_t const * input, float * output)
#endif
{
  int nthread = __MOPSR_UNPACK_TPB;
  int nblocks = ndat / nthread;

  // each thread will unpack 1 complex time sample from 1 channel
  dim3 blocks (nblocks, nchan);

#ifdef _DEBUG
  cerr << "mopsr_unpack_fpt ndat=" << ndat << " nchan=" << nchan
       << " input=" << (void*) input << " output=" << (void *) output 
       << " nblocks=" << nblocks << " nthread=" << nthread << endl;
#endif

  float2 * complex_output = (float2 *) output;

  const unsigned npol = 1;
  const unsigned ndim = 2;

  // for output order FTP
  const unsigned chan_stride = ndat * npol;

  // for input order TFP
  const unsigned samp_stride = nchan * npol * ndim;

#ifdef  USE_TEXTURE_MEMORY
  //mopsr_unpack_complex_1<<<nblock,nthread,0,stream>>>(complex_output, *tex);
#else
  mopsr_unpack_fpt_complex_1<<<blocks,nthread,0,stream>>>(input, complex_output, chan_stride, samp_stride);
#endif

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error_stream ("mopsr_unpack_fpt", stream);
}

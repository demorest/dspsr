//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2013 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <stdio.h>

#include "dsp/MOPSRUnpackerCUDA.h"
#include "dsp/Operation.h"
#include "debug.h"
#include "Error.h"

// threads per block - C1060=256 [TODO CHECK below if changing]
#define __MOPSR_UNPACK_TPB 1024

#define __MOPSR_SAMPLES_PER_THREAD 4

#define WARP_SIZE 32

// global static texture declaration for MOPSR gpu unpacker
//texture<char2, cudaTextureType1D, cudaReadModeNormalizedFloat> mopsr_tex1dfloat2;

// real textutre version
//texture<int8_t, cudaTextureType1D, cudaReadModeElementType> mopsr_tex1dfloat;

using namespace std;

//void check_error (const char*);
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
                                            float * output,
                                            const unsigned nval,
                                            const unsigned nchan, 
                                            const unsigned nsamp_per_block,
                                            const unsigned chan_stride)
{
  extern __shared__ int8_t sdata[];
  const unsigned ndim = 2;

  // input index
  const unsigned idx = (blockIdx.x * blockDim.x + threadIdx.x);
  const unsigned in_idx = idx * ndim;

  // shared memory index
  const unsigned sin_idx = threadIdx.x * ndim;


  if (idx >= nval)
  {
    sdata[sin_idx] = 0;
    sdata[sin_idx+1] = 0;
  }
  else
  {
    sdata[sin_idx]   = stagingBufGPU[in_idx];
    sdata[sin_idx+1] = stagingBufGPU[in_idx+1];
  }

  // synchronize all threads in the block
  __syncthreads(); 

  // now we have 1000 consective (complex) TF samples in the sdata array stored as int8_t

  // determine the output index for this thread
  const unsigned ichan    = threadIdx.x / nsamp_per_block;
  const unsigned isamp    = threadIdx.x % nsamp_per_block;

  // determine which shared memory index for this output ichan and isamp
  const unsigned sout_idx = ((isamp * nchan) + ichan) * ndim;

  // convert to float
  float re = (float) sdata[sout_idx];
  float im = (float) sdata[sout_idx+1];

  // + 0.5 since scale is -128 to 127
  re += 0.5;
  im += 0.5;

  // optimal scaling from bit table
  re *= mopsr_unpacker_scale;
  im *= mopsr_unpacker_scale;

  // finally determine the output index for this thread
  const unsigned ou_idx = (ichan * chan_stride) + (blockIdx.x * nsamp_per_block * ndim) + (isamp * ndim);

  if (blockIdx.x * nsamp_per_block * nchan < nval)
  {
#if _KDEBUG
    if (blockIdx.x == 0)
      printf ("threadIdx.x=%d sin_idx=%d, ichan=%d, isamp=%d, sout_idx=%d, ou_idx=%d\n", threadIdx.x, sin_idx, ichan, isamp, sout_idx, ou_idx);
#endif

    output[ou_idx]   = re;
    output[ou_idx+1] = im;
  }
  else
  {
    printf ("blockIdx.x=%d, threadIdx.x=%d val=%d >= nval=%d\n", blockIdx.x, threadIdx.x, blockIdx.x * nsamp_per_block * nchan, nval);
  }

#if _KDEBUG
  if (blockIdx.x ==0 && threadIdx.x == 0)
    printf ("=========================\n");
#endif
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

void mopsr_unpack_prepare (cudaStream_t stream, const float scale)
{
  cudaError_t error = cudaMemcpyToSymbolAsync ( mopsr_unpacker_scale, &scale, sizeof(scale), 0, cudaMemcpyHostToDevice, stream);
  // TODO check return value
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
  const unsigned npol = 1;
  const unsigned ndim = 2;
  const unsigned nval = ndat * nchan;

  // we want the number of threads to be module nchan
  int nthread = (__MOPSR_UNPACK_TPB / nchan) * nchan;
  int nblocks = nval / nthread;
  if (nval % nthread)
    nblocks++;

  // each thread will unpack 1 complex time sample from 1 channel
  size_t sdata_bytes = nthread * ndim;
  const unsigned nsamp_per_block = nthread/nchan;
  const unsigned chan_stride = ndat * npol * ndim;

  if (dsp::Operation::verbose)
    cerr << "mopsr_unpack_fpt nval=" << nval << " ndat=" << ndat << " nchan=" << nchan
         << " input=" << (void*) input << " output=" << (void *) output 
         << " nblocks=" << nblocks << " nthread=" << nthread
         << " sdata_bytes=" << sdata_bytes << " nsamp_per_block=" << nsamp_per_block
         << " chan_stride=" << chan_stride << endl;

#ifdef  USE_TEXTURE_MEMORY
  //mopsr_unpack_complex_1<<<nblock,nthread,0,stream>>>(complex_output, *tex);
#else
  mopsr_unpack_fpt_complex_1<<<nblocks,nthread,sdata_bytes,stream>>>(input, output, nval, nchan, nsamp_per_block, chan_stride);
#endif

  // AJ's theory... 
  // If there are no stream synchronises on the input then the CPU pinned memory load from the
  // input class might be able to get ahead of a whole sequence of GPU operations, and even exceed
  // one I/O loop. Therefore this should be a reuqirement to have a stream synchronize some time
  // after the data are loaded from pinned memory to GPU ram and the next Input copy to pinned memory

  // put it here for now
  cudaStreamSynchronize(stream);

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error_stream ("mopsr_unpack_fpt", stream);
}

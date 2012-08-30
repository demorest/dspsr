//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/CASPSRUnpackerCUDA.h"
#include "dsp/Operation.h"

#include "Error.h"

// threads per block - C1060=256 [TODO CHECK below if changing]
#define __CASPSR_UNPACK_TPB 256

// global static texture declaration for CASPSR gpu unpacker
texture<int8_t, 1, cudaReadModeElementType> caspsr_unpack_tex;

using namespace std;

/* 
   Unpack the two real-valued input polarizations into an interleaved
   array suited to the twofft algorithm described in Section 12.3
   of Numerical Recipes
*/

typedef struct { int8_t val[8]; } char8;

#define convert(s,i) (float(i)+0.5f)*s

__global__ void unpack_real_ndim2 (uint64_t ndat, const float scale,
				   const char8* input, float* output)
{
  uint64_t index = blockIdx.x*blockDim.x + threadIdx.x;
  output += index * 8;
 
  output[0] = convert(scale,input[index].val[0]);
  output[1] = convert(scale,input[index].val[4]);
  output[2] = convert(scale,input[index].val[1]);
  output[3] = convert(scale,input[index].val[5]);
  output[4] = convert(scale,input[index].val[2]);
  output[5] = convert(scale,input[index].val[6]);
  output[6] = convert(scale,input[index].val[3]);
  output[7] = convert(scale,input[index].val[7]);
}

void check_error (const char*);

#ifdef USE_TEXTURE_MEMORY
// ndim 1 unpacker uses texture memory for reads
__global__ void unpack_real_ndim1 (float* into_pola, float* into_polb, float scale)
{
  const int idx                 = blockIdx.x*blockDim.x + threadIdx.x;
  const int sample_idx          = idx * 8;
  unsigned int shared_idx       = threadIdx.x * 4;
  const uint64_t output_idx     = blockIdx.x * blockDim.x * 4;
  const unsigned int half_block = blockDim.x / 2;

  // n.b. this is blockDim.x * 4 [hardcoded by default]
  __shared__ float pola[4 * __CASPSR_UNPACK_TPB];
  __shared__ float polb[4 * __CASPSR_UNPACK_TPB];

  // loads 8 samples per thread (4 per poln)
  unsigned i = 0;

  // write 4 samples from each poln into shared memory
  for (i=0; i<4; i++)
  {

    pola[shared_idx + i] = (((float) tex1Dfetch(caspsr_unpack_tex, sample_idx + i)) + 0.5) * scale;
    polb[shared_idx + i] = (((float) tex1Dfetch(caspsr_unpack_tex, sample_idx + i + 4)) + 0.5) * scale;
  }

  __syncthreads();

  // first half threads write poln A
  if (threadIdx.x < half_block)
  {
    unsigned int tid = 2 * threadIdx.x + (48 * ((int) (threadIdx.x/8)));
    float * to = into_pola + output_idx;

    to[tid + 0]  = pola[tid + 0];
    to[tid + 1]  = pola[tid + 1];
    to[tid + 16] = pola[tid + 16];
    to[tid + 17] = pola[tid + 17];
    to[tid + 32] = pola[tid + 32];
    to[tid + 33] = pola[tid + 33];
    to[tid + 48] = pola[tid + 48];
    to[tid + 49] = pola[tid + 49];
  }
  // second half threads write poln B
  else
  {
    unsigned int tid = 2 * (threadIdx.x - half_block) + (48 * ((int) ((threadIdx.x-half_block)/8)));
    float * to = into_polb + output_idx;

    to[tid + 0]  = polb[tid + 0];
    to[tid + 1]  = polb[tid + 1];
    to[tid + 16] = polb[tid + 16];
    to[tid + 17] = polb[tid + 17];
    to[tid + 32] = polb[tid + 32];
    to[tid + 33] = polb[tid + 33];
    to[tid + 48] = polb[tid + 48];
    to[tid + 49] = polb[tid + 49];
  }
}
#else
__global__ void unpack_real_ndim1 (uint64_t ndat, float scale,
				   const unsigned char* stagingBufGPU,
				   float* into_pola, float* into_polb) 
{
  uint64_t sampleTmp = blockIdx.x*blockDim.x + threadIdx.x; 

  uint64_t outputIndex = sampleTmp * 4;
  sampleTmp = sampleTmp * 8;
 
  float* to_A = into_pola + outputIndex;
  float* to_B = into_polb + outputIndex;

  const int8_t* from = reinterpret_cast<const int8_t*>( stagingBufGPU ) + sampleTmp;

  to_A[0] = ((float) from[0] + 0.5) * scale;
  to_A[1] = ((float) from[1] + 0.5) * scale;
  to_A[2] = ((float) from[2] + 0.5) * scale;
  to_A[3] = ((float) from[3] + 0.5) * scale;

  to_B[0] = ((float) from[4] + 0.5) * scale;
  to_B[1] = ((float) from[5] + 0.5) * scale;
  to_B[2] = ((float) from[6] + 0.5) * scale;
  to_B[3] = ((float) from[7] + 0.5) * scale;
}
#endif

void caspsr_unpack (cudaStream_t stream, const uint64_t ndat, float scale, 
                    unsigned char const* input, float* pol0, float* pol1)
{
  int nthread = __CASPSR_UNPACK_TPB;

  // each thread will unpack 4 time samples from each polarization
  int nblock = ndat / (4*nthread);

#ifdef _DEBUG
  cerr << "caspsr_unpack ndat=" << ndat << " scale=" << scale 
       << " input=" << (void*) input << " nblock=" << nblock
       << " nthread=" << nthread << endl;
#endif

#ifdef USE_TEXTURE_MEMORY
  unpack_real_ndim1<<<nblock,nthread,0,stream>>> (pol0, pol1, scale);
#else
  unpack_real_ndim1<<<nblock,nthread,0,stream>>> (ndat, scale, input, pol0, pol1);
#endif

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error ("caspsr_unpack");
}

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

#ifdef USE_TEXTURE_MEMORY
__global__ void mopsr_unpack_complex_1 (float2 * output, cudaTextureObject_t tex)
{
  const int idx = blockIdx.x*blockDim.x + threadIdx.x;
  output[idx] = tex1Dfetch<float2>(tex, idx);
}
#else
__global__ void mopsr_unpack_complex_1 (float scale, 
                                           const unsigned char* stagingBufGPU,
                                           float2* output)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  const int8_t* from = reinterpret_cast<const int8_t*>( stagingBufGPU ) + idx;

  __shared__ float2 out;

  out.x = ((float) from[idx + 0] + 0.5) * scale;
  out.y = ((float) from[idx + 1] + 0.5) * scale;

  output[idx] = out;
}
#endif

#ifdef USE_TEXTURE_MEMORY
void mopsr_unpack (cudaStream_t stream, const uint64_t ndat,
                      unsigned char const* input, float * output,
                      cudaTextureObject_t * tex)
#else
void mopsr_unpack (cudaStream_t stream, const uint64_t ndat,
                      float scale, unsigned char const* input, float * output)
#endif
{
  int nthread = __MOPSR_UNPACK_TPB;

  // each thread will unpack 4 complex time samples
  int nblock = ndat / nthread;

#ifdef _DEBUG
  cerr << "mopsr_unpack ndat=" << ndat
       << " input=" << (void*) input << " nblock=" << nblock
       << " nthread=" << nthread << endl;
#endif

  float2 * complex_output = (float2 *) output;

#ifdef  USE_TEXTURE_MEMORY
  mopsr_unpack_complex_1<<<nblock,nthread,0,stream>>>(complex_output, *tex);
#else
  mopsr_unpack_complex_1<<<nblock,nthread,0,stream>>>(scale, input, complex_output);
#endif

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error_stream ("mopsr_unpack", stream);
}


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


__global__ void unpack_real_ndim1 (uint64_t ndat, float scale,
				   const unsigned char* stagingBufGPU,
				   float* into_pola, float* into_polb) 
{
  uint64_t sampleIndex,sampleTmp;
  char sample;
  uint64_t outputIndex;

  sampleTmp = blockIdx.x*blockDim.x + threadIdx.x; 

  outputIndex = sampleTmp * 4;
  sampleTmp = sampleTmp * 8;
 
  float* to_A = into_pola + outputIndex;
  float* to_B = into_polb + outputIndex;
  const int8_t* from_A = reinterpret_cast<const int8_t*>( stagingBufGPU ) + sampleTmp;
  const int8_t* from_B = from_A + 4;
  
  //  unsigned nunpack = 4;
    /* if (outputIndex + 4 > ndat) 
  {
    if ((ndat - outputIndex > 0) && (ndat - outputIndex < 4))
      nunpack = ndat - outputIndex;
    else
      nunpack = 0;
      }*/

  for (unsigned i=0; i<4; i++)
  {
    // ensure that mean is zero then scale so that variance is unity
    to_A[i] = ((float) from_A[i] + 0.5) * scale; 
    to_B[i] = ((float) from_B[i] + 0.5) * scale; 
  }
}


void caspsr_unpack (const uint64_t ndat, float scale, 
                    unsigned char const* input, float* pol0, float* pol1)
{
  int nthread = 256;

  // each thread will unpack 4 time samples from each polarization
  int nblock = ndat / (4*nthread);

#ifdef _DEBUG
    cerr << "caspsr_unpack ndat=" << ndat << " scale=" << scale 
         << " input=" << (void*) input << " nblock=" << nblock
         << " nthread=" << nthread << endl;
#endif

#if 0
  unpack_real_ndim2<<<nblock, nthread>>> (ndat, scale,
					  (const char8*) input, pol0);
#else
  unpack_real_ndim2<<<nblock, nthread>>> (ndat, scale,
					  input, pol0, pol1);
#endif

  if (dsp::Operation::record_time)
  {
    cudaThreadSynchronize ();
 
    cudaError error = cudaGetLastError();
    if (error != cudaSuccess)
      throw Error (InvalidState, "caspsr_unpack", cudaGetErrorString (error));
  }
}


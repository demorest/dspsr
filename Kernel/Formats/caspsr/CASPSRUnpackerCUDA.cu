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

__global__ void unpack_real_npol2 (uint64_t ndat, const float scale,
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

void caspsr_unpack (const uint64_t ndat, float scale, 
                    unsigned char const* input, float* output)
{
  int nthread = 256;

  // each thread will unpack 4 time samples from each polarization
  int nblock = ndat / (4*nthread);

#ifdef _DEBUG
    cerr << "caspsr_unpack ndat=" << ndat << " scale=" << scale 
         << " input=" << (void*) input << " nblock=" << nblock
         << " nthread=" << nthread << endl;
#endif

  unpack_real_npol2<<<nblock, nthread>>> (ndat, scale,
                   reinterpret_cast<const char8*>(input), output);

  if (dsp::Operation::record_time)
  {
    cudaThreadSynchronize ();
 
    cudaError error = cudaGetLastError();
    if (error != cudaSuccess)
      throw Error (InvalidState, "caspsr_unpack", cudaGetErrorString (error));
  }
}


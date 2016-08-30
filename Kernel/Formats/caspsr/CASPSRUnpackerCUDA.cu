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

void check_error_stream (const char*, cudaStream_t);

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
				   int8_t * from, float* into_pola, float* into_polb) 
{
  extern __shared__ int8_t sdata[];

  unsigned idx_shm = threadIdx.x;
  unsigned idx     = (8 * blockIdx.x * blockDim.x) + threadIdx.x;
  unsigned i;

  // each thread will load 8 values (coalesced) from GMEM to SHM
  for (i=0; i<8; i++)
  {
    if (idx < 2*ndat)
    {
      sdata[idx_shm] = from[idx];

      idx     += blockDim.x;
      idx_shm += blockDim.x;
    }
  }

  __syncthreads();

  idx     = (4 * blockIdx.x * blockDim.x) + threadIdx.x;
  idx_shm = threadIdx.x + ((threadIdx.x / 4) * 4);

  // each thread will write 4 values (coalesced) from SHM to GMEM
  for (i=0; i<4; i++)
  {
    if (idx < ndat)
    {
      into_pola[idx] = ((float) sdata[idx_shm]   + 0.5) * scale; 
      into_polb[idx] = ((float) sdata[idx_shm+4] + 0.5) * scale;

      idx += blockDim.x;
      idx_shm += blockDim.x * 2;
    }
  }
}

void caspsr_unpack (cudaStream_t stream, const uint64_t ndat, float scale, 
                    unsigned char const* input, float* pol0, float* pol1,
                    int nthread)
{

  // each thread will unpack 4 time samples from each polarization
  int nsamp_per_block = 4 * nthread;
  int nblock = ndat / nsamp_per_block;
  if (ndat % nsamp_per_block)
    nblock++;

#ifdef _DEBUG
  cerr << "caspsr_unpack ndat=" << ndat << " scale=" << scale 
       << " input=" << (void*) input << " nblock=" << nblock
       << " nthread=" << nthread << endl;
#endif

  int8_t * from = (int8_t *) input;
  size_t shm_bytes = 8 * nthread;
  unpack_real_ndim1<<<nblock,nthread,shm_bytes,stream>>> (ndat, scale, from, pol0, pol1);

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error_stream ("caspsr_unpack", stream);
}

//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2009
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/CASPSRUnpackerCUDA.h"

#include "Error.h"
#include <cutil_inline.h>


using namespace std;

typedef float2 Complex;

const float UPPER_BITS 0xff000000;
const float MID_BITS 0xff0000;
const float LOW_BITS 0xff00;
const float BOTTOM_BITS 0xff;

//static float stagingBufGPU; 
//static float unpackBufGPU;

__global__ void unpackDataGPU(float*, float*,
			      int, float, float,
			      float, float);


dsp::CASPSRUnpackerGPU::CASPSRUnpackerGPU (const char* _name) 
{
  if (verbose)
    cerr << "dsp::CASPSRUnpacker ctor" << endl;
}

dsp::CASPSRUnpackerGPU::~CASPSRUnpackerGPU ()
{

}

void dsp::CASPSRUnpackerGPU::unpack()
{
  
  
  // in theory we already know what all of these values are...

  //const unsigned nchan = input->get_nchan();
  //const unsigned ndim  = input->get_ndim();
  const uint64_t ndat  = input->get_ndat();
  
    
  int dimBlockUnpack(512);
  //int dataSize = nchan * bwd_nfft;// need to get hold of bwd_nfft
  unsigned dataSize = ndat;

  int dimGridUnpack(dataSize / dimBlockUnpack); // must be less than 33Mpt. Otherwise should be ok. (Max thread per block 512, max grid dim 65535)

  //int packedDim = (nchan * bwd_nfft)/4 ;
  unsigned packedDim = ndat/4;
  unsigned halfData = dataSize / 2;

  // buffer on host - should already be allocated

  // buffer on gpu for packed data
  cudaSafeCall(cudaMalloc((void **) &stagingBufGPU,packedDim*sizeof(cufftReal)));

  cutilSafeCall(cudaMemcpy(stagingBufGPU,host_mem,packedDim*sizeof(cufftReal),cudaMemcpyHostToDevice));

  // buffer on gpu for unpacked data
  cudaSafeCall(cudaMalloc((void **) &unpackBufGPU,packedDim*sizeof(cufftReal)*4));

  // mem cpy data from cpu to gpu staging buf
 
  unpackDataGPU<<<dimGridUnpack,dimBlockUnpack>>>(stagingBufGPU,unpackBufGPU, halfData, UPPER_BITS, MID_BITS, LOW_BITS, BOTTOM_BITS);
}

/* unpack the byte data into float format on the GPU.
   The input is a sequence of 4 8-bit numbers for 1 pol, then 4 8-bit numbers for the next
*/
__global__ void unpackDataGPU(float *stagingBufGPU, float *unpackBufGPU,
			      int halfData, float UPPER_BITS, float MID_BITS,
			      float LOW_BITS, float BOTTOM_BITS) 
{
  int outputIndex,sampleIndex;
  float sample;
  

  // output will also be based on pol. if sample index is odd... output index must
  // be put after midpoint of data

  sampleIndex = blockIdx.x*blockDim.x + threadIdx.x; 
  
  if (sampleIndex & 0x01)
    {
      outputIndex= (sampleIndex-1)*2 + halfData;
    }
  else
    {
      outputIndex= sampleIndex*2;
    }
  
  sample = stagingBufGPU[sampleIndex];
  // needs to be shifted by 24 bits
  unpackBufGPU[outputIndex] = (sample&UPPER_BITS)>>24;
  // needs to be shifted by 16 bits 
  unpackBufGPU[outputIndex+1] = (sample&MID_BITS)>>16;
  // needs to be shifted by 8 bits 
  unpackBufGPU[outputIndex+2] =(sample&LOW_BITS)>>8;
  // is fine 
  unpackBufGPU[outputIndex+3] = sample&BOTTOM_BITS; 
}

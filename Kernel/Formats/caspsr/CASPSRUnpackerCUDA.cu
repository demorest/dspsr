//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2010
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/CASPSRUnpackerCUDA.h"

#include "Error.h"

using namespace std;

typedef float2 Complex;

/* unpack the byte data into float format on the GPU.
   The input is a sequence of 4 8-bit numbers for 1 pol, then 4 8-bit numbers for the next
*/

__global__ void unpackDataCUDA(uint64_t ndat, float scale,
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
  
  return; 
  /*
  sampleIndex= sampleTmp;
  if (sampleIndex > ndat*2)
	return;
  sample = stagingBufGPU[sampleIndex];
  into_pola[outputIndex]= (float) sample;
 
  sampleIndex = sampleTmp + 1;
  if (sampleIndex > ndat*2)
	return;
  sample = stagingBufGPU[sampleIndex];
  into_pola[outputIndex+1]= (float) sample;
   
  sampleIndex = sampleTmp + 2;
  if (sampleIndex > ndat*2)
	return;
  sample = stagingBufGPU[sampleIndex];
  into_pola[outputIndex+2]= (float) sample;
 
  sampleIndex = sampleTmp + 3;
  if (sampleIndex > ndat*2)
	return;
  sample = stagingBufGPU[sampleIndex];
  into_pola[outputIndex+3]= (float) sample;
 
  sampleIndex = sampleTmp + 4;
 // if (sampleIndex > ndat*2)
 //	return;
  sample = stagingBufGPU[sampleIndex];
  into_polb[outputIndex]= (float) sample;
    
  sampleIndex = sampleTmp + 5;
 //if (sampleIndex > ndat*2)
 //	return;
  sample = stagingBufGPU[sampleIndex];
  into_polb[outputIndex+1]= (float) sample;
    
  sampleIndex = sampleTmp + 6;
 // if (sampleIndex > ndat*2)
 //	return;
  sample = stagingBufGPU[sampleIndex];
  into_polb[outputIndex+2]= (float) sample;
    
  sampleIndex = sampleTmp + 7;
 // if (sampleIndex > ndat*2)
 //	return;
  sample = stagingBufGPU[sampleIndex];
  into_polb[outputIndex+3]= (float) sample;
   */
 
}

void caspsr_unpack(const uint64_t ndat, float scale, unsigned char const* stagingBufGPU,int dimBlockUnpack,int dimGridUnpack,float* into_pola, float* into_polb)
{
  //cerr << "dimGrid: " << dimGridUnpack << " dimBlock: " << dimBlockUnpack << endl;

  unpackDataCUDA<<<dimGridUnpack,dimBlockUnpack>>>(ndat,scale,stagingBufGPU, into_pola,into_polb);

  cudaThreadSynchronize ();
  cudaError error = cudaGetLastError();
  if (error != cudaSuccess)
    cerr << "FAIL: " << cudaGetErrorString (error) << endl;
}


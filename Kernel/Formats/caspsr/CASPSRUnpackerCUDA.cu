//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2010
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/CASPSRUnpackerCUDA.h"
//#include "dsp/BitTable.h"

#include "Error.h"

//#include <cutil_inline.h>


using namespace std;

typedef float2 Complex;

//const uint32_t UPPER_BITS=0xff000000;
//const uint32_t MID_BITS=0xff0000;
//const uint32_t LOW_BITS=0xff00;
//const uint32_t BOTTOM_BITS=0xff;


/* unpack the byte data into float format on the GPU.
   The input is a sequence of 4 8-bit numbers for 1 pol, then 4 8-bit numbers for the next
*/

__global__ void unpackDataCUDA(const unsigned char* stagingBufGPU,
			       unsigned halfData, float* into_pola, float* into_polb) 
{
  unsigned int sampleIndex,sampleTmp;
  unsigned char sample;

  // output will also be based on pol. if sample index is odd... output index must
  // be put after midpoint of data

  sampleTmp = blockIdx.x*blockDim.x + threadIdx.x; 
  sampleTmp = sampleTmp * 8;
  

  sampleIndex= sampleTmp;
  sample = stagingBufGPU[sampleIndex];
  into_pola[sampleIndex]=sample;

  sampleIndex = sampleTmp + 1;
  sample = stagingBufGPU[sampleIndex];
  into_pola[sampleIndex]=sample;

  sampleIndex = sampleTmp + 2;
  sample = stagingBufGPU[sampleIndex];
  into_pola[sampleIndex]=sample;

  sampleIndex = sampleTmp + 3;
  sample = stagingBufGPU[sampleIndex];
  into_pola[sampleIndex]=sample;

  sampleIndex = sampleTmp + 4;
  sample = stagingBufGPU[sampleIndex];
  //  unpackBufGPU[sampleIndex+halfData]=sample;
  into_polb[sampleIndex]=sample;

  sampleIndex = sampleTmp + 5;
  sample = stagingBufGPU[sampleIndex];
  into_polb[sampleIndex]=sample;

  sampleIndex = sampleTmp + 6;
  sample = stagingBufGPU[sampleIndex];
  into_polb[sampleIndex]=sample;

  sampleIndex = sampleTmp + 7;
  sample = stagingBufGPU[sampleIndex];
  into_polb[sampleIndex]=sample;



  /*  if (sampleIndex & 0x01)
    {
      outputIndex= (sampleIndex-1)*2 + halfData;
    }
  else
    {
      outputIndex= sampleIndex*2;
    }
  
    sample = stagingBufGPU[sampleIndex];*/

  // unpacker assumed 32-bit numbers - will actually be getting as 8-bit.
  // needs to be shifted by 24 bits
  //  unpackBufGPU[outputIndex] = (sample&UPPER_BITS)>>24;
  // needs to be shifted by 16 bits 
  //unpackBufGPU[outputIndex+1] = (sample&MID_BITS)>>16;
  // needs to be shifted by 8 bits 
  //unpackBufGPU[outputIndex+2] =(sample&LOW_BITS)>>8;
  // is fine 
  //unpackBufGPU[outputIndex+3] = sample&BOTTOM_BITS; 
}


void caspsr_unpack(const uint64_t ndat, unsigned char* stagingBufGPU,int dimBlockUnpack,int dimGridUnpack,unsigned halfData, float* into_pola, float* into_polb)
{
    
  unpackDataCUDA<<<dimGridUnpack,dimBlockUnpack>>>(stagingBufGPU, halfData,into_pola,into_polb);

}


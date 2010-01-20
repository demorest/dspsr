//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2010
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/CASPSRUnpackerSetup.h"
#include "dsp/BitTable.h"
#include "dsp/CASPSRUnpackerCUDA.h"
#include "Error.h"
#include <cutil_inline.h>
#include <cufft.h>
using namespace std;

dsp::CASPSRUnpackerSetup::CASPSRUnpackerSetup (const char* _name) : HistUnpacker (_name)
{
}


dsp::CASPSRUnpackerSetup::~CASPSRUnpackerSetup ()
{
}

void dsp::CASPSRUnpackerSetup::unpack()
{
    
  // in theory we already know what all of these values are...
  //const uint64_t ndat  = input->get_ndat();
  
  const uint64_t ndat = input->get_ndat();
  uint64_t dataSize = ndat;
  int dimBlockUnpack(512);
  int dimGridUnpack(dataSize / dimBlockUnpack*8); // must be less than 33Mpt. Otherwise should be ok. (Max thread per block 512, max grid dim 65535)

  //int packedDim = (nchan * bwd_nfft)/4 ;
  unsigned packedDim = ndat/4;
  unsigned halfData = dataSize / 2;

  // buffer on host - should already be allocated

  // buffer on gpu for packed data
  cutilSafeCall(cudaMalloc((void **) &stagingBufGPU,packedDim*sizeof(cufftReal)));

  cutilSafeCall(cudaMemcpy(stagingBufGPU,host_mem,packedDim*sizeof(cufftReal),cudaMemcpyHostToDevice));

  // buffer on gpu for unpacked data
  cutilSafeCall(cudaMalloc((void **) &unpackBufGPU,packedDim*sizeof(cufftReal)*4));

  //call function
  caspsr_unpack(ndat,host_mem,stagingBufGPU,unpackBufGPU,dimBlockUnpack,dimGridUnpack,halfData);
}


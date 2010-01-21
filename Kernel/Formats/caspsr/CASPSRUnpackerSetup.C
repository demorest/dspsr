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

  // staging buffer on the GPU for packed data
  //unsigned char* stagingBufGPU; -- should not be necessary, cause it'll be
  //                              -- directly pointed to by get_rawptr();

  // buffer for unpacked data on the GPU
  //float* unpackBufGPU; -- replaced by "output->get_datptr"
  
  //unsigned char* host_mem; -- replaced by "input->get_rawptr()"
  
  
  const uint64_t ndat = input->get_ndat();
  const unsigned char* stagingBufGPU = input->get_rawptr();
  float* into_pola = output->get_datptr(0,0);
  float* into_polb = output->get_datptr(0,1);


  //uint64_t dataSize = ndat;
  int dimBlockUnpack(512);
  int dimGridUnpack(ndat / dimBlockUnpack*8); 

  //int packedDim = (nchan * bwd_nfft)/4 ;
  //unsigned packedDim = ndat/4;
  unsigned halfData = ndat / 2;

  // buffer on host - should already be allocated

  // buffer on gpu for packed data
  //cutilSafeCall(cudaMalloc((void **) &stagingBufGPU,packedDim*sizeof(cufftReal)));

  //cutilSafeCall(cudaMemcpy(stagingBufGPU,from,packedDim*sizeof(cufftReal),cudaMemcpyHostToDevice));

  // buffer on gpu for unpacked data
  // should be malloc earlier along with the stagin buf? just need the pointers getdatprt? 
  //cutilSafeCall(cudaMalloc((void **) &unpackBufGPU,ndat*sizeof(cufftReal)));

  //call function
  caspsr_unpack(ndat,stagingBufGPU,dimBlockUnpack,dimGridUnpack,halfData,into_pola,into_polb);
}


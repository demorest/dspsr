/*

 */

#ifndef __dsp_GenericEightBitUnpackerCUDA_h
#define __dsp_GenericEightBitUnpackerCUDA_h

#include<stdint.h>
#include<cuda_runtime.h>

#include "dsp/BitTable.h"

struct unpack_dimensions
{
  uint64_t ndat;
  unsigned nchan;
  unsigned npol;
  unsigned ndim;
};

void generic_8bit_unpack (cudaStream_t stream, 
			  const unpack_dimensions& dim,
			  const dsp::BitTable*,
			  const unsigned char* input,
			  float* output, uint64_t stride);

#endif

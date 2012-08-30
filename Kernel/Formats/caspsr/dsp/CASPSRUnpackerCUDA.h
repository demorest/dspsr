/*

 */

#ifndef __dsp_CASPSRUnpackerCUDA_h
#define __dsp_CASPSRUnpackerCUDA_h

#define USE_TEXTURE_MEMORY 1

#include<stdint.h>
#include<cuda_runtime.h>

void caspsr_texture_alloc (void * d_staging, size_t size);

void caspsr_unpack (cudaStream_t stream, const uint64_t ndat,
		    float scale,
		    const unsigned char* stagingBufGPU,
		    float* pol0, float* pol1);

#endif

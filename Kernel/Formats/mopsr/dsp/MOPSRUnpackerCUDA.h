/***************************************************************************
 *
 *   Copyright (C) 2013 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_MOPSRUnpackerCUDA_h
#define __dsp_MOPSRUnpackerCUDA_h

//#define USE_TEXTURE_MEMORY 

#include <stdint.h>
#include <cuda_runtime.h>

#include <dsp/MemoryCUDA.h>

void mopsr_texture_alloc (void * d_staging, size_t size);

void mopsr_unpack_prepare (cudaStream_t stream, const float scale);

#ifdef USE_TEXTURE_MEMORY
void mopsr_unpack (cudaStream_t stream, const uint64_t ndat,
                   const unsigned char* stagingBufGPU,
                   float* into, cudaTextureObject_t * tex);
#else
void mopsr_unpack_fpt (cudaStream_t stream, const uint64_t ndat, const unsigned nchan,
                       float scale, int8_t const * input, float * output);
#endif
void mopsr_unpack_tfp (cudaStream_t stream, const uint64_t ndat, const unsigned nchan,
                       float scale, int8_t const * input, float * output);

#endif


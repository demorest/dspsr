/*

 */

#ifndef __dsp_KAT7UnpackerCUDA_h
#define __dsp_KAT7UnpackerCUDA_h

#include<stdint.h>
#include<cuda_runtime.h>

void kat7_unpack (cudaStream_t stream, const uint64_t ndat, unsigned nchan, unsigned npol,
                  float scale, const int16_t * input, float * output);

#endif

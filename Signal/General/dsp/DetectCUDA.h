
#ifndef __dsp_DetectCUDA_h
#define __dsp_DetectCUDA_h

#include<stdint.h>

void polarimetryCUDA (int BlkThread, int Blks,unsigned nchan, unsigned sdet, const float* p, const float* q, uint64_t ndat, unsigned ndim, float* S0, float* S2);

#endif

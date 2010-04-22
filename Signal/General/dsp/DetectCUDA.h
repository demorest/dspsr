
#ifndef __dsp_DetectCUDA_h
#define __dsp_DetectCUDA_h

#include<stdint.h>

void polarimetry_ndim4 (float* base, uint64_t span, 
			uint64_t ndat, unsigned nchan);

void polarimetry_ndim2 (float* base, uint64_t span, 
			uint64_t ndat, unsigned nchan);

#endif

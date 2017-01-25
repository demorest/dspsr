/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
// dspsr/Signal/General/cross_detect.h

#ifndef __cross_detect_h
#define __cross_detect_h

#ifdef __cplusplus
extern "C" {
#endif

void cross_detect (unsigned ndat, const float* p, const float* q,
		   float* pp, float* qq, float* Rpq, float* Ipq, 
                   unsigned span);

void cross_detect_int (unsigned ndat, const float* p, const float* q,
		       float* pp, float* qq, float* Rpq, float* Ipq, 
		       unsigned span);

#ifdef __cplusplus
}
#endif

#endif


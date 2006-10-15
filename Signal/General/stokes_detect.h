/***************************************************************************
 *
 *   Copyright (C) 2004 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
/* $Source: /cvsroot/dspsr/dspsr/Signal/General/stokes_detect.h,v $
   $Revision: 1.1 $
   $Date: 2006/10/15 19:09:24 $
   $Author: straten $ */

#ifndef __stokes_detect_h
#define __stokes_detect_h

#ifdef __cplusplus
extern "C" {
#endif

void stokes_detect (unsigned ndat, const float* p, const float* q,
		    float* S0, float* S1, float* S2, float* S3, 
		    unsigned span);

void stokes_detect_int (unsigned ndat, const float* p, const float* q,
			float* S0, float* S1, float* S2, float* S3, 
			unsigned span);

#ifdef __cplusplus
}
#endif

#endif


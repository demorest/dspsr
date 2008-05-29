/***************************************************************************
 *
 *   Copyright (C) 2004 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "stokes_detect.h"

#define CD_OP =
void stokes_detect (unsigned ndat, const float* p, const float* q,
		   float* S0, float* S1, float* S2, float* S3, 
                   unsigned span)
#include "stokes_detect.ic"

#undef CD_OP
#define CD_OP +=
void stokes_detect_int (unsigned ndat, const float* p, const float* q,
		       float* S0, float* S1, float* S2, float* S3, 
		       unsigned span)
#include "stokes_detect.ic"

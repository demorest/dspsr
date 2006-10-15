/***************************************************************************
 *
 *   Copyright (C) 2002 by pulsar Swinburne University
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "cross_detect.h"

#define CD_OP =
void cross_detect (unsigned ndat, const float* p, const float* q,
		   float* pp, float* qq, float* Rpq, float* Ipq, 
                   unsigned span)
#include "cross_detect.ic"

#undef CD_OP
#define CD_OP +=
void cross_detect_int (unsigned ndat, const float* p, const float* q,
		       float* pp, float* qq, float* Rpq, float* Ipq, 
		       unsigned span)
#include "cross_detect.ic"

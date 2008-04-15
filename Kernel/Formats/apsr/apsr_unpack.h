//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __APSR_unpack_h
#define __APSR_unpack_h

#include "dsp/BitUnpacker.h"

void apsr_unpack (const dsp::BitSeries* input, dsp::TimeSeries* output,
                  dsp::BitUnpacker* unpacker);

#endif


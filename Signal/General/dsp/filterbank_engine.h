//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/filterbank_engine.h

#ifndef __filterbank_engine_h
#define __filterbank_engine_h

/*
  The nvcc compiler in CUDA release version 4.0 has a bug 

  http://forums.nvidia.com/index.php?showtopic=210798

  that causes it to fail when compiling Transformation.h

  This C struct can be used to decouple a filterbank implementation
  from the (standard) C++ used by the dsp::Filterbank::Engine.
*/

typedef struct
{
  float* scratch;

  float* output;
  unsigned output_span;

  unsigned nchan;
  unsigned nchan_subband;
  unsigned freq_res;
  unsigned nfilt_pos;
  unsigned nkeep;
}
  filterbank_engine;

#endif


//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011 by Willem van Straten and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __baseband_cuda_SKMasker_h
#define __baseband_cuda_SKMasker_h

#include "dsp/SKMasker.h"
#include "dsp/MemoryCUDA.h"

namespace CUDA
{
  class SKMaskerEngine : public dsp::SKMasker::Engine
  {
  public:

    //! Default Constructor
    SKMaskerEngine (dsp::Memory * memory);

    void setup ();

    void perform (dsp::BitSeries* mask, const dsp::TimeSeries* input,
                  dsp::TimeSeries* out, unsigned M);

  protected:

    DeviceMemory * device_memory;

    cudaStream_t stream;

    int max_threads_per_block;

  };
}

#endif


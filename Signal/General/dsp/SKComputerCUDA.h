//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2016 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __baseband_cuda_SKComputer_h
#define __baseband_cuda_SKComputer_h

#include "dsp/SKComputer.h"
#include "dsp/MemoryCUDA.h"

namespace CUDA
{
  class Memory;

  class SKComputerEngine : public dsp::SKComputer::Engine
  {
  public:

    //! Default Constructor
    SKComputerEngine (dsp::Memory * memory);

    void setup ();

    void compute (const dsp::TimeSeries* input, dsp::TimeSeries* output,
                  dsp::TimeSeries *output_tscr, unsigned tscrunch);

    void insertsk (const dsp::TimeSeries* input, dsp::TimeSeries* output,
                   unsigned tscrunch);

  protected:

    DeviceMemory * device_memory;

    cudaStream_t stream;

    // device work buffer
    float * work_buffer;

    size_t work_buffer_size;

    int max_threads_per_block;

  };
}

#endif

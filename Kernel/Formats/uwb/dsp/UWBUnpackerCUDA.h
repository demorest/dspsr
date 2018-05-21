//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2014 by Andrew JAmeson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __baseband_cuda_UWBUnpacker_h
#define __baseband_cuda_UWBUnpacker_h

#include "dsp/UWBUnpacker.h"
#include <cuda_runtime.h>

namespace CUDA
{
  class UWBUnpackerEngine : public dsp::UWBUnpacker::Engine
  {
  public:

    //! Default Constructor
    UWBUnpackerEngine (cudaStream_t stream);

    void setup ();

    bool get_device_supported (dsp::Memory* memory) const;

    void set_device (dsp::Memory* memory);

    void unpack (const dsp::BitSeries * input, dsp::TimeSeries * output);

  protected:

    cudaStream_t stream;

    struct cudaDeviceProp gpu;

    dsp::BitSeries staging;

    bool first_block;

  };
}

#endif

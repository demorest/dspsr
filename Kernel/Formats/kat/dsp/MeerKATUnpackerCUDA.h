/***************************************************************************
 *
 *   Copyright (C) 2015 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_MeerKATUnpackerCUDA_h
#define __dsp_MeerKATUnpackerCUDA_h

#include "dsp/MeerKATUnpacker.h"

#include <cuda_runtime.h>

namespace CUDA
{

  class MeerKATUnpackerEngine : public dsp::MeerKATUnpacker::Engine
  {
  public:

    //! Default Constructor
    MeerKATUnpackerEngine (cudaStream_t stream);

    void setup ();

    bool get_device_supported (dsp::Memory* memory) const;

    void set_device (dsp::Memory* memory);

    void unpack (float scale, const dsp::BitSeries * input, dsp::TimeSeries * output, unsigned sample_swap);

  protected:

    cudaStream_t stream;

    struct cudaDeviceProp gpu;

    dsp::BitSeries staging;

  };
}


#endif

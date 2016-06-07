//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2014 by Andrew JAmeson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __baseband_cuda_SKA1Unpacker_h
#define __baseband_cuda_SKA1Unpacker_h

#include <cuda_runtime.h>

//#include "dsp/SKA1Unpacker.h"
#ifdef SKA1_ENGINE_IMPLEMENTATION
namespace CUDA
{
  class SKA1UnpackerEngine : public dsp::SKA1Unpacker::Engine
  {
  public:

    //! Default Constructor
    SKA1UnpackerEngine (cudaStream_t stream);

    void setup ();

    bool get_device_supported (dsp::Memory* memory) const;

    void set_device (dsp::Memory* memory);

    void unpack (float scale, const dsp::BitSeries * input, dsp::TimeSeries * output);

  protected:

    cudaStream_t stream;

    struct cudaDeviceProp gpu;

    dsp::BitSeries staging;

  };
}
#else

#include <inttypes.h>

void ska1_unpack_tfp (cudaStream_t stream, uint64_t nval, float scale,
                      float * into, void * staged,
                      unsigned  nchan, unsigned npol, unsigned ndim,
                      size_t pol_span);

void ska1_unpack_fpt (cudaStream_t stream, uint64_t ndat, float scale,
                      float * into, void * staged, unsigned  nchan,
                      size_t pol_span);
#endif

#endif

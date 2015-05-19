//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2015 by Matthew Kerr
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __FZoomEngine_h
#define __FZoomEngine_h

#include "dsp/FZoom.h"

#include <cuda_runtime.h>

namespace CUDA
{
  class FZoomEngine : public dsp::FZoom::Engine
  {
  public:

    FZoomEngine (cudaStream_t stream);
    //~FZoomEngine ();

    void fpt_copy (const dsp::TimeSeries * input, 
                         dsp::TimeSeries * output,
                         unsigned chan_lo,
                         unsigned chan_hi);

  protected:

    cudaStream_t stream;

    cudaMemcpyKind get_kind ();
  };
}

#endif

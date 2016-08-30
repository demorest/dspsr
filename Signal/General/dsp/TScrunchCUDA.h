//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2015 by Matthew Kerr
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __TScrunchEngine_h
#define __TScrunchEngine_h

#include "dsp/TScrunch.h"

#include <cuda_runtime.h>

namespace CUDA
{
  class TScrunchEngine : public dsp::TScrunch::Engine
  {
  public:

    TScrunchEngine (cudaStream_t stream);

    void fpt_tscrunch (const dsp::TimeSeries * input, 
                       dsp::TimeSeries * output,
                       unsigned sfactor);

  protected:

    cudaStream_t stream;
  };
}

#endif // !defined(__TScrunchEngine_h)

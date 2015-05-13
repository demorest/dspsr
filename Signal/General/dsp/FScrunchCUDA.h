//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2015 by Matthew Kerr
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __FScrunchEngine_h
#define __FScrunchEngine_h

#include "dsp/FScrunch.h"

#include <cuda_runtime.h>

namespace CUDA
{
  class FScrunchEngine : public dsp::FScrunch::Engine
  {
  public:

    FScrunchEngine (cudaStream_t stream);

    void fpt_fscrunch (const dsp::TimeSeries * input, 
                         dsp::TimeSeries * output,
                         unsigned sfactor);

  protected:

    cudaStream_t stream;
  };
}

#endif // !defined(__FScrunchEngine_h)

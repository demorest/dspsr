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

namespace CUDA
{
  class SKMaskerEngine : public dsp::SKMasker::Engine
  {
  public:

    //! Default Constructor
    SKMaskerEngine (cudaStream_t stream);

    void setup (unsigned nchan, unsigned npol, unsigned span);

    void perform (dsp::BitSeries* mask, unsigned mask_offset, dsp::TimeSeries* out, 
                  unsigned offset, unsigned end);

  protected:
    cudaStream_t stream;

    unsigned nchan;

    unsigned npol;

    //! DDFB span, i.e. n floats between channels from raw base ptr
    unsigned span;

    int max_threads_per_block;

  };
}

#endif


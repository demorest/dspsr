//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2010 by Jonathon Kocz and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __baseband_cuda_FourierDigitizer_h
#define __baseband_cuda_FourierDigitizer_h

#include "dsp/FourierDigitizer.h"

namespace CUDA
{
  class FourierDigitizerEngine : public dsp::FourierDigitizer::Engine
  {
  public:
    //! Default Constructor
    FourierDigitizerEngine (cudaStream_t stream);

    void pack (int nbit, const dsp::TimeSeries* in, dsp::BitSeries* out);
      
    void finish ();

  protected:
    cudaStream_t stream;

  };
}

#endif


//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011 by Willem van Straten and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __baseband_cuda_ZapWeight_h
#define __baseband_cuda_ZapWeight_h

#include "dsp/ZapWeight.h"

namespace CUDA
{
  class ZapWeightEngine : public dsp::ZapWeight::Engine
  {
  public:

    //! Default Constructor
    ZapWeightEngine (cudaStream_t stream);
    ~ZapWeightEngine ();

    void setup ();
    void perform ();

    void polarimetry (unsigned ndim,
                      const dsp::TimeSeries* in, dsp::TimeSeries* out);

  protected:
    cudaStream_t stream;

  };
}

#endif


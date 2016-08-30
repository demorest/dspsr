//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2015 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __baseband_cuda_PolnSelect_h
#define __baseband_cuda_PolnSelect_h

#include "dsp/PolnSelect.h"
#include "dsp/LaunchConfig.h"

namespace CUDA
{
  class PolnSelectEngine : public dsp::PolnSelect::Engine
  {
  public:

    //! Default Constructor
    PolnSelectEngine (cudaStream_t stream);

    ~PolnSelectEngine ();

    void setup ();

    void fpt_polnselect (int ipol, 
                         const dsp::TimeSeries* in,
                         dsp::TimeSeries* out);

    void tfp_polnselect (int ipol,
                         const dsp::TimeSeries* in,
                         dsp::TimeSeries* out);

  protected:

    cudaStream_t stream;

    //! gpu configuration
    LaunchConfig gpu_config; 

  };
}

#endif


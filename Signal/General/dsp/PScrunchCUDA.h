//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2015 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __baseband_cuda_PScrunch_h
#define __baseband_cuda_PScrunch_h

#include "dsp/PScrunch.h"
#include "dsp/LaunchConfig.h"

namespace CUDA
{
  class PScrunchEngine : public dsp::PScrunch::Engine
  {
  public:

    //! Default Constructor
    PScrunchEngine (cudaStream_t stream);

    ~PScrunchEngine ();

    void setup ();

    void fpt_pscrunch (const dsp::TimeSeries* in,
                       dsp::TimeSeries* out);

    void tfp_pscrunch (const dsp::TimeSeries* in,
                       dsp::TimeSeries* out);

  protected:

    cudaStream_t stream;

    //! gpu configuration
    LaunchConfig gpu_config; 

  };
}

#endif


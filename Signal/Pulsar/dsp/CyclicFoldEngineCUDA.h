//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __baseband_cuda_CyclicFold_h
#define __baseband_cuda_CyclicFold_h

#include "dsp/CyclicFold.h"

namespace CUDA
{

  class CyclicFoldEngineCUDA : public dsp::CyclicFoldEngine
  {
  public:

    CyclicFoldEngineCUDA (cudaStream_t stream);
    ~CyclicFoldEngineCUDA ();

    void set_ndat (uint64_t ndat, uint64_t idat_start);

    void fold ();

    void zero ();

    void synch (dsp::PhaseSeries *);

  protected:

    // Copy of binplan on device
    unsigned* d_binplan[2];

    // Send the binplan to GPU
    void send_binplan ();

    // memory for temp results on the device
    float* d_lagdata;

    // Get the lag data from GPU
    // TODO the lag->profile conversion could be done on GPU
    void get_lagdata ();

    cudaStream_t stream;
  };
}

#endif

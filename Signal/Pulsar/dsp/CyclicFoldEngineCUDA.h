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
#include "dsp/FoldCUDA.h"

namespace CUDA
{

  class CyclicFoldEngineCUDA : public dsp::CyclicFoldEngine
  {
  public:

    CyclicFoldEngineCUDA (cudaStream_t stream);
    ~CyclicFoldEngineCUDA ();
    
    void set_bin (uint64_t idat, double ibin, double bins_per_samp);
    uint64_t set_bins (double phi, double phase_per_sample, uint64_t _ndat, uint64_t idat_start);
    uint64_t get_bin_hits (int ibin);

    void set_ndat (uint64_t ndat, uint64_t idat_start);

    void fold ();

    void zero ();

    void synch (dsp::PhaseSeries *);

  protected:
    // Rather than bothering with binplan inherited from CyclicFold, we make
    // our own bin-lag indexed binplan with a 2-d array. Dimensions will be [bin][lag]
    CUDA :: bin *lagbinplan;
    // Copy of binplan on device
    CUDA :: bin *d_binplan;

    // Send the binplan to GPU
    void send_binplan ();

    // memory for temp results on the device
    float* d_lagdata;

    unsigned current_turn;
    unsigned last_ibin;

    // Get the lag data from GPU
    // TODO the lag->profile conversion could be done on GPU
    void get_lagdata ();

    cudaStream_t stream;
  };
}

#endif

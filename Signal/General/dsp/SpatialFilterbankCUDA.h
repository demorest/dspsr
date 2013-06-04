//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2012 by Willem van Straten and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __SpatialFilterbankCUDA_h
#define __SpatialFilterbankCUDA_h

#include "dsp/FilterbankEngine.h"
#include "dsp/SpatialFilterbank.h"

#include <cufft.h>

namespace CUDA
{
  //! Filterbank step implemented using CUDA streams
  class SpatialFilterbankEngine : public dsp::Filterbank::Engine
  {
    unsigned nstream;

  public:

    //! Default Constructor
    SpatialFilterbankEngine (cudaStream_t stream);

    ~SpatialFilterbankEngine ();

    void setup (dsp::Filterbank*);
    void set_scratch (float *);
    void create_plan ();
    void create_batched_plan (uint64_t npart, unsigned npol, uint64_t in_step, uint64_t out_step);
    void perform (const dsp::TimeSeries* in, dsp::TimeSeries* out,
                  uint64_t npart, uint64_t in_step, uint64_t out_step);
    void finish ();

    int max_threads_per_block; 
  private:

    //! forward fft plan 
    cufftHandle plan_fwd;

    bool plan_prepared;

    //! Complex-valued data
    bool real_to_complex;

    //! number of FFTs to perform in batch mode
    unsigned int nbatch;

    //! inplace FFT in CUDA memory
    float2* d_fft;

    cudaStream_t stream;

    //! nfft / nantenna (X dimension)
    unsigned int nx;

    //! nchan (Y dimension)
    unsigned int ny;

  };

}

#endif

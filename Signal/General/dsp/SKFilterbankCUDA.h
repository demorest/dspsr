//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2015 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __baseband_cuda_SKFilterbank_h
#define __baseband_cuda_SKFilterbank_h

#include <cufft.h>
#include <config.h>

#include "dsp/SKFilterbank.h"
#include "dsp/MemoryCUDA.h"
#include "dsp/LaunchConfig.h"

namespace CUDA
{
  class SKFilterbankEngine : public dsp::SKFilterbank::Engine
  {
  public:

    //! Default Constructor
    SKFilterbankEngine (dsp::Memory * _memory, unsigned _tscrunch);

    ~SKFilterbankEngine();

    void setup ();

    void prepare (const dsp::TimeSeries* input, unsigned _nfft);

    void perform (const dsp::TimeSeries* input,
                  dsp::TimeSeries* output,
                  dsp::TimeSeries* output_tscr);

  protected:

    DeviceMemory * memory;

    void fft_real (cufftReal *in, cufftComplex * out);

    void fft_complex (cufftComplex *in, cufftComplex * out);

    cudaStream_t stream;

    cufftType type;

    cufftHandle plan;

    void * buffer;

    size_t buffer_size;

    void * sums;

    size_t sums_size;

    int nchan;

    int npol;

    int npt;

    int nbatch;

    int tscrunch;
 
    int max_threads_per_block;
  };
}

#endif


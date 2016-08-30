//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2016 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __baseband_cuda_SpectralKurtosis_h
#define __baseband_cuda_SpectralKurtosis_h

#include "dsp/SpectralKurtosis.h"

#include "dsp/MemoryCUDA.h"
#include "dsp/SKComputerCUDA.h"
#include "dsp/SKDetectorCUDA.h"
#include "dsp/SKMaskerCUDA.h"

#include "dsp/TransferCUDA.h"
#include "dsp/TransferBitSeriesCUDA.h"

namespace CUDA
{

  class SpectralKurtosisEngine : public dsp::SpectralKurtosis::Engine
  {
  public:

    //! Default Constructor
    SpectralKurtosisEngine (dsp::Memory * memory);

    void setup ();

    void compute (const dsp::TimeSeries* input, dsp::TimeSeries* output,
                  dsp::TimeSeries *output_tscr, unsigned tscrunch);

    void reset_mask (dsp::BitSeries* output);

    void detect_ft (const dsp::TimeSeries* input, dsp::BitSeries* output,
                    float upper_thresh, float lower_thresh);

    void detect_fscr (const dsp::TimeSeries* input, dsp::BitSeries* output,
                      const float lower, const float upper,
                      unsigned schan, unsigned echan);

    void detect_tscr (const dsp::TimeSeries* input,
                      const dsp::TimeSeries * input_tscr,
                      dsp::BitSeries* output,
                      float upper, float lower);

    int count_mask (const dsp::BitSeries* output);

    float * get_estimates (const dsp::TimeSeries * estimates);

    unsigned char * get_zapmask (const dsp::BitSeries * zapmask);

    void mask (dsp::BitSeries* mask, const dsp::TimeSeries *in, dsp::TimeSeries* out, unsigned M);

    void insertsk (const dsp::TimeSeries* input, dsp::TimeSeries* out, unsigned M);

  protected:

    DeviceMemory * device_memory;

    cudaStream_t stream;

    SKComputerEngine * computer;

    SKDetectorEngine * detector;

    SKMaskerEngine * masker;

    float * work_buffer;

    size_t work_buffer_size;

    int max_threads_per_block;

  };
}

#endif

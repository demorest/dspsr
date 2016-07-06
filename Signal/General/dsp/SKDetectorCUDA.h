//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2016 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __baseband_cuda_SKDetector_h
#define __baseband_cuda_SKDetector_h

#include "dsp/SKDetector.h"
#include "dsp/MemoryCUDA.h"

#include "dsp/TransferCUDA.h"
#include "dsp/TransferBitSeriesCUDA.h"

namespace CUDA
{
  class SKDetectorEngine : public dsp::SKDetector::Engine
  {
  public:

    //! Default Constructor
    SKDetectorEngine (dsp::Memory * memory);

    void setup ();

    void reset_mask (dsp::BitSeries* output);

    void detect_ft (const dsp::TimeSeries* input, dsp::BitSeries* output,
                    float upper_thresh, float lower_thresh);

    void detect_fscr(const dsp::TimeSeries* input, dsp::BitSeries* output, 
                     const float lower, const float upper,
                     unsigned schan, unsigned echan);

    void detect_tscr (const dsp::TimeSeries* input, const dsp::TimeSeries* input_tscr, dsp::BitSeries* output, 
                      float upper_thresh, float lower_thresh);

    int count_mask (const dsp::BitSeries* output);

    float * get_estimates (const dsp::TimeSeries* input);

    unsigned char * get_zapmask (const dsp::BitSeries* input);


  protected:

    DeviceMemory * device_memory;

    cudaStream_t stream;

    unsigned nchan;

    unsigned npol;

    //! DDFB span, i.e. n floats between channels from raw base ptr
    unsigned span;

    int max_threads_per_block;

    PinnedMemory * pinned_memory;

    dsp::TimeSeries * estimates_host;

    dsp::BitSeries * zapmask_host;

    dsp::TransferCUDA* transfer_estimates;

    dsp::TransferBitSeriesCUDA* transfer_zapmask;

  };
}

#endif

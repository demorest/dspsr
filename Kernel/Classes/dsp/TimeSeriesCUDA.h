//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2016 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __TimeSeriesEngine_h
#define __TimeSeriesEngine_h

#include "dsp/TimeSeries.h"
#include "dsp/MemoryCUDA.h"

#include <cuda_runtime.h>

namespace CUDA
{
  class TimeSeriesEngine : public dsp::TimeSeries::Engine
  {
  public:

    //! Default constructor
    TimeSeriesEngine (dsp::Memory * _memory);

    //! Copy constructor
    //TimeSeriesEngine (const TimeSeriesEngine& tse);

    ~TimeSeriesEngine ();

    //TimeSeriesEngine& operator = (const TimeSeriesEngine& copy);

    void prepare (dsp::TimeSeries * parent);

    void prepare_buffer (unsigned nbytes);

    void copy_data_fpt (const dsp::TimeSeries * copy,
                        uint64_t idat_start = 0,
                        uint64_t ndat = 0);

    void copy_data_fpt_same_stream (const dsp::TimeSeries * from,
            uint64_t idat_start, uint64_t ndat);

    void copy_data_fpt_same_device (const dsp::TimeSeries * from,
            uint64_t idat_start, uint64_t ndat);

    void copy_data_fpt_diff_device (const dsp::TimeSeries * from,
            uint64_t idat_start, uint64_t ndat);

    void copy_data_fpt_kernel_multidim (float * to, const float * from,
            uint64_t to_stride, uint64_t from_stride, 
            uint64_t idat_start, uint64_t ndat, cudaStream_t stream);

    void * buffer;

  protected:

    dsp::TimeSeries * to;

    CUDA::DeviceMemory * memory;

    CUDA::PinnedMemory * pinned_memory;

    void * host_buffer;

    size_t host_buffer_size;

    size_t buffer_size;

    unsigned nchan;

    unsigned npol;

    unsigned ndim;

    uint64_t ichanpol_stride;

    uint64_t ochanpol_stride;

    uint64_t bchanpol_stride;

    unsigned nthread;

    dim3 blocks;

    int device;

    cudaStream_t to_stream;
      
    cudaStream_t from_stream;

    int to_device;

    int from_device;

  };
}

#endif // !defined(__TimeSeriesEngine_h)

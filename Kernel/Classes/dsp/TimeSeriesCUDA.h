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

    void * buffer;

  protected:

    dsp::TimeSeries * to;

    CUDA::DeviceMemory * memory;

    size_t buffer_size;

  };
}

#endif // !defined(__TimeSeriesEngine_h)

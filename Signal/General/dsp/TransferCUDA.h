//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __TransferCUDA_h
#define __TransferCUDA_h

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"

#include <cuda_runtime.h>

namespace dsp {

  class TransferCUDA : public Transformation<TimeSeries,TimeSeries>
  {
  public:

    //! Default constructor - always out of place
    TransferCUDA(cudaStream_t _stream);

    void set_kind (cudaMemcpyKind k) { kind = k; }
    void prepare ();
    
    // If transferring all input in its own stream, need the stream and an event
    // signaling transfer completion
    void set_input_stream (cudaStream_t _input_stream, cudaEvent_t _event);

    Operation::Function get_function () const { return Operation::Structural; }

  protected:

    //! Do stuff
    void transformation();

    cudaMemcpyKind kind;

    cudaStream_t stream;

    cudaStream_t input_stream;

    cudaEvent_t event;

  };

}

#endif

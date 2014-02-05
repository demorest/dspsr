//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __TransferBitSeriesCUDA_h
#define __TransferBitSeriesCUDA_h

#include "dsp/Transformation.h"
#include "dsp/BitSeries.h"

#include <cuda_runtime.h>

namespace dsp {

  class TransferBitSeriesCUDA : public Transformation<BitSeries,BitSeries>
  {
  public:

    //! Default constructor - always out of place
    TransferBitSeriesCUDA(cudaStream_t _stream);

    void set_kind (cudaMemcpyKind k) { kind = k; }
    void prepare ();

    Operation::Function get_function () const { return Operation::Structural; }

  protected:

    //! Do stuff
    void transformation();

    cudaMemcpyKind kind;

    cudaStream_t stream;

  };

}

#endif

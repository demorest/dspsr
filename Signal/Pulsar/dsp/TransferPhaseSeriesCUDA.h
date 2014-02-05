//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __TransferPhaseSeriesCUDA_h
#define __TransferPhaseSeriesCUDA_h

#include "dsp/Transformation.h"
#include "dsp/PhaseSeries.h"

#include <cuda_runtime.h>

namespace dsp {

  class TransferPhaseSeriesCUDA : public Transformation<PhaseSeries,PhaseSeries>
  {
  public:

    //! Default constructor - always out of place
    TransferPhaseSeriesCUDA(cudaStream_t _stream);

    void set_kind (cudaMemcpyKind k) { kind = k; }
    void prepare ();
    void set_transfer_hits (bool transfer) { transfer_hits = transfer; }

    Operation::Function get_function () const { return Operation::Structural; }

  protected:

    //! Do stuff
    void transformation();

    cudaMemcpyKind kind;

    bool transfer_hits;

    cudaStream_t stream;
  };

}

#endif

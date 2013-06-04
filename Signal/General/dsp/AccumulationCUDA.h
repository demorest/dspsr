//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2010 by Jonathon Kocz and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/AccumulationCUDA.h,v $
   $Revision: 1.3 $
   $Date: 2010/06/01 10:46:29 $
   $Author: straten $ */

#ifndef __baseband_cuda_Accumulation_h
#define __baseband_cuda_Accumulation_h

#include "dsp/Accumulation.h"

namespace CUDA
{
  class AccumulationEngine : public dsp::Accumulation::Engine
  {
  public:

    //! Default Constructor
    AccumulationEngine (cudaStream_t stream);

    void integrate (const dsp::TimeSeries* in, dsp::TimeSeries* out,
                    unsigned tscrunch, unsigned stride);


  protected:
    cudaStream_t stream;

  };
}

#endif


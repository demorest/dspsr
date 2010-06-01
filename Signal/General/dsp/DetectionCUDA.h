//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2010 by Jonathon Kocz and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/DetectionCUDA.h,v $
   $Revision: 1.3 $
   $Date: 2010/06/01 10:46:29 $
   $Author: straten $ */

#ifndef __baseband_cuda_Detection_h
#define __baseband_cuda_Detection_h

#include "dsp/Detection.h"

namespace CUDA
{
  class DetectionEngine : public dsp::Detection::Engine
  {
  public:

    //! Default Constructor
    DetectionEngine (cudaStream_t stream);

    void polarimetry (unsigned ndim,
                      const dsp::TimeSeries* in, dsp::TimeSeries* out);

  protected:
    cudaStream_t stream;

  };
}

#endif


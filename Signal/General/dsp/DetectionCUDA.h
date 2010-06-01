//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2010 by Jonathon Kocz and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/DetectionCUDA.h,v $
   $Revision: 1.1 $
   $Date: 2010/06/01 08:50:29 $
   $Author: straten $ */

#ifndef __baseband_cuda_Detection_h
#define __baseband_cuda_Detection_h

#include "dsp/Detection.h"

namespace CUDA
{
  class DetectionEngine : public dsp::Detection::Engine
  {
  public:
    void polarimetry (unsigned ndim, const TimeSeries* in, TimeSeries* out);
  };
}

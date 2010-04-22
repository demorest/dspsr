//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2010 by Jonathon Kocz and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/FoldCUDA.h,v $
   $Revision: 1.1 $
   $Date: 2010/04/22 05:40:47 $
   $Author: straten $ */

#ifndef __baseband_cuda_Fold_h
#define __baseband_cuda_Fold_h

#include "dsp/Fold.h"

namespace CUDA
{
  class FoldEngine : public dsp::Fold::Engine
  {
  public:

    void set_binplan (uint64_t ndat, unsigned* bins);
    void fold ();

  protected:

    unsigned* binplan_ptr;
    uint64_t binplan_size;
  };

#endif

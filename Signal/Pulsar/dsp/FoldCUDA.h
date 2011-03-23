//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2010 by Jonathon Kocz and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/FoldCUDA.h,v $
   $Revision: 1.10 $
   $Date: 2011/03/23 19:42:24 $
   $Author: straten $ */

#ifndef __baseband_cuda_Fold_h
#define __baseband_cuda_Fold_h

#include "dsp/Fold.h"
#include "dsp/TransferCUDA.h"

namespace CUDA
{
  typedef struct 
  {
    unsigned ibin;
    unsigned hits;
    uint64_t offset;
  } bin;

  class FoldEngine : public dsp::Fold::Engine
  {
  public:

    FoldEngine (cudaStream_t stream);
    ~FoldEngine ();

    void set_nbin (unsigned nbin);
    void set_ndat (uint64_t ndat, uint64_t idat_start);

    void set_bin (uint64_t idat, double ibin, double unused=0);

    void fold ();

    dsp::PhaseSeries* get_profiles ();

    void synch (dsp::PhaseSeries*);

    void zero () { get_profiles()->zero(); }

  protected:

    bin* binplan;
    uint64_t binplan_size;
    uint64_t binplan_nbin;

    unsigned current_bin;
    unsigned current_hits;
    unsigned folding_nbin;

    void send_binplan ();

    bin* d_bin;
    uint64_t d_bin_size;

    // memory for profiles on the device
    Reference::To<dsp::PhaseSeries> d_profiles;

    // operation used to transfer data from device to host
    Reference::To<dsp::TransferCUDA> transfer;

    cudaStream_t stream;
  };
}

#endif

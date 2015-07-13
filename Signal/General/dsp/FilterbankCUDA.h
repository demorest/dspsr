//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/FilterbankCUDA.h,v $
   $Revision: 1.17 $
   $Date: 2011/10/07 11:10:14 $
   $Author: straten $ */

#ifndef __FilterbankCUDA_h
#define __FilterbankCUDA_h

#include "dsp/FilterbankEngine.h"
#include "dsp/LaunchConfig.h"

#include <cufft.h>

namespace CUDA
{
  class elapsed
  {
  public:
    elapsed ();
    void wrt (cudaEvent_t before);

    double total;
    cudaEvent_t after;
  };

  //! Discrete convolution filterbank step implemented using CUDA streams
  class FilterbankEngine : public dsp::Filterbank::Engine
  {
    unsigned nstream;
    
  public:

    //! Default Constructor
    FilterbankEngine (cudaStream_t stream);

    ~FilterbankEngine ();

    void setup (dsp::Filterbank*);
    void set_scratch (float *);

    void perform (const dsp::TimeSeries* in, dsp::TimeSeries* out,
                  uint64_t npart, uint64_t in_step, uint64_t out_step);

    void finish ();

  protected:

    //! forward fft plan 
    cufftHandle plan_fwd;

    //! backward fft plan
    cufftHandle plan_bwd;

    //! Complex-valued data
    bool real_to_complex;

    //! inplace FFT in CUDA memory
    float2* d_fft;

    //! convolution kernel in CUDA memory
    float2* d_kernel;

    //! device scratch sapce
    float* scratch;

    unsigned nchan_subband;
    unsigned freq_res;
    unsigned nfilt_pos;
    unsigned nkeep;

    LaunchConfig1D multiply;

    cudaStream_t stream;

    bool verbose;

  };

}

#endif

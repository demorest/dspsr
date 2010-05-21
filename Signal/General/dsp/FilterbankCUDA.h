//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/FilterbankCUDA.h,v $
   $Revision: 1.14 $
   $Date: 2010/05/21 00:00:51 $
   $Author: straten $ */

#ifndef __FilterbankCUDA_h
#define __FilterbankCUDA_h

#include "dsp/Filterbank.h"

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
  class Engine : public dsp::Filterbank::Engine
  {
    unsigned nstream;

  public:

    //! Default Constructor
    Engine ();
    ~Engine ();

    void setup (dsp::Filterbank*);
    void perform (const float* in);
    bool dual_poln () { return twofft; }
    void finish ();

  protected:

    //! forward fft plan 
    cufftHandle plan_fwd;
    //! backward fft plan
    cufftHandle plan_bwd;

    //! the backward fft length
    unsigned bwd_nfft;

    bool real_to_complex;

    //! Use the twofft trick from NR
    bool twofft;

    //! inplace FFT in CUDA memory
    float2* d_fft;
    //! convolution kernel in CUDA memory
    float2* d_kernel;

    //! real-to-complex trick arrays in CUDA memory
    float *d_SN, *d_CN;

  };

}

#endif

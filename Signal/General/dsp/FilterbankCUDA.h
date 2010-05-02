//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/FilterbankCUDA.h,v $
   $Revision: 1.12 $
   $Date: 2010/05/02 17:58:44 $
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

    //! Adds the streams
    void setup (unsigned nchan, unsigned bwd_nfft, float* kernel);
    void perform (const float* in);
    bool dual_poln () { return twofft; }
    void finish ();

  private:

    float* kernel;

    //! Initializes CUDA stream-specific resources
    void init ();

  protected:
  
    //! forward fft plan 
    cufftHandle plan_fwd;
    //! backward fft plan
    cufftHandle plan_bwd;

    //! the backward fft length
    unsigned bwd_nfft;

    //! Use the twofft trick from NR
    bool twofft;

    //! inplace FFT in CUDA memory
    float2* d_fft;
    //! convolution kernel in CUDA memory
    float2* d_kernel;

    //! real-to-complex trick arrays in CUDA memory
    float *d_SN, *d_CN;

  private:

    bool built;

  };

}

#endif

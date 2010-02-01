//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/FilterbankCUDA.h,v $
   $Revision: 1.8 $
   $Date: 2010/02/01 07:19:01 $
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

  private:

    float* kernel;

    //! Initializes CUDA stream-specific resources
    void init ();

  protected:

    //! stream identifier
    //cudaStream_t stream;
    
    //! forward fft plan 
    cufftHandle plan_fwd;
    //! backward fft plan
    cufftHandle plan_bwd;

    //! the backward fft length
    unsigned bwd_nfft;
    //! the number of frequency channels produced by filterbank
    unsigned nchan;

    //! inplace FFT in CUDA memory
    float2* d_fft;
    //! convolution kernel in CUDA memory
    float2* d_kernel;

    //! pinned memory on host used for asynchronous memcpy to GPU
    float* pinned;

    //! real-to-complex trick arrays in CUDA memory
    float *d_SN, *d_CN;

  private:

    cudaEvent_t start;

    class Timers
    {
    public:
      elapsed copy_to;
      elapsed fwd;    
      elapsed realtr; 
      elapsed conv;   
      elapsed bwd;    
      elapsed copy_from;
    };

    Timers* timers;
  };

}

#endif

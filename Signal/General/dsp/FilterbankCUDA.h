//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/FilterbankCUDA.h,v $
   $Revision: 1.4 $
   $Date: 2009/11/15 00:51:17 $
   $Author: straten $ */

#ifndef __FilterbankCUDA_h
#define __FilterbankCUDA_h

#include "dsp/Filterbank.h"

#include <cufft.h>
#include <cutil_inline.h>

namespace CUDA
{
  //! Discrete convolution filterbank step implemented using CUDA streams
  class Filterbank : public dsp::Filterbank::Engine
  {
    unsigned nstream;

  public:

    //! Construct with number of streams and install CUDA::Memory policy
    Filterbank (unsigned _nstream);

    //! Manages stream-specific resources
    class Stream;

    //! Adds the streams
    void setup (unsigned nchan, unsigned bwd_nfft, float* kernel);

    //! Starts the data reduction steps for all streams
    void run ();

    Stream* get_stream (unsigned i);
  };

  class Filterbank::Stream : public QuasiMutex::Stream
  {
  private:
    const Stream* copy;
    float* kernel;

    //! Sets some attributes to zero
    void zero ();

    //! Initializes CUDA stream-specific resources
    void init ();

    //! Initializes CUDA resources that are shared between streams
    void work_init ();

    //! Copies CUDA resources from the copy attribute
    void copy_init ();

  protected:

    //! stream identifier
    cudaStream_t stream;
    
    //! forward fft plan 
    cufftHandle plan_fwd;
    //! backward fft plan
    cufftHandle plan_bwd;

    //! the backward fft length
    unsigned bwd_nfft;
    //! the number of frequency channels produced by filterbank
    unsigned nchan;

    //! input data in CUDA memory
    float2* d_in;
    //! output data in CUDA memory
    float2* d_out;
    //! convolution kernel in CUDA memory
    float2* d_kernel;

    //! pinned memory on host used for asynchronous memcpy to GPU
    float* pinned;

    //! real-to-complex trick arrays in CUDA memory
    float *d_SN, *d_CN;
        
    friend class Filterbank;
    
    void forward_fft ();
    void realtr ();
    void convolve ();
    void backward_fft ();
    void retrieve ();
    
  public:
    
    Stream (unsigned nchan, unsigned bwd_nfft, float* kernel);
    Stream (const Stream* copy);
    
    void queue () ;
    void run () ;
    void wait () ;
  };

}

#endif

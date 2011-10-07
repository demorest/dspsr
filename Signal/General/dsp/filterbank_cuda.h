//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/filterbank_cuda.h,v $
   $Revision: 1.1 $

   $Date: 2011/10/07 11:10:14 $
   $Author: straten $ */

#ifndef __filterbank_cuda_h
#define __filterbank_cuda_h

#include "dsp/filterbank_engine.h"
#include <cufft.h>

/*
  The nvcc compiler in CUDA release version 4.0 has a bug 

  http://forums.nvidia.com/index.php?showtopic=210798

  that causes it to fail when compiling Transformation.h

  This C struct is used to decouple the CUDA::FilterbankEngine
  implementation from the (standard) C++ used by the 
  dsp::Filterbank::Engine.
*/

typedef struct
{
  //! forward fft plan 
  cufftHandle plan_fwd;

  //! backward fft plan
  cufftHandle plan_bwd;

  //! the backward fft length
  unsigned bwd_nfft;

  //! Complex-valued data
  bool real_to_complex;

  //! Use the twofft trick from NR
  bool twofft;

  //! inplace FFT in CUDA memory
  float2* d_fft;

  //! convolution kernel in CUDA memory
  float2* d_kernel;

  //! real-to-complex trick arrays in CUDA memory
  float *d_SN, *d_CN;

  cudaStream_t stream;

  bool verbose;
}
  filterbank_cuda;

void filterbank_cuda_perform (filterbank_engine* engine, 
			      filterbank_cuda* cuda,
			      const float* in);
#endif


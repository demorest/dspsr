//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2015 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __baseband_cuda_Convolution_Callbacks_h
#define __baseband_cuda_Convolution_Callbacks_h

#include <cufft.h>
#include <config.h>

#if HAVE_CUFFT_CALLBACKS

  void setup_callbacks_ConvolutionCUDA (cufftHandle plan_fwd, cufftHandle plan_bwd,
																			  cufftHandle plan_fwd_batch, cufftHandle plan_bwd_batch,
																			  cufftComplex * d_kernels, int nbatch, cudaStream_t stream);

  void setup_callbacks_conv_params (unsigned * h_ptr, unsigned h_size, cudaStream_t stream);

  void setup_callbacks_ConvolutionCUDASpectral (cufftHandle plan_fwd, cufftHandle plan_bwd,
                                     					  cufftComplex * d_kernels, cudaStream_t stream);

  void setup_callbacks_conv_params_spectral (unsigned * h_ptr, unsigned h_size, cudaStream_t stream);

#endif

#endif //__baseband_cuda_Convolution_Callbacks_h

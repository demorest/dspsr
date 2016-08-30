//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2015 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/ConvolutionCUDACallbacks.h"
#include "CUFFTError.h"
#include "debug.h"

#if HAVE_CUFFT_CALLBACKS
#include <cufftXt.h>
#endif

using namespace std;

#if HAVE_CUFFT_CALLBACKS

// [0] channel offset ( ichan * npt)
// [1] npt
// [2] first_ipt ( nfilt_pos )
// [3] last_ipt ( npt - nfilt_neg )
// [4] nfilt_tot 
__device__ __constant__ unsigned conv_params[5];

/////////////////////////////////////////////////////////////////////////
//
// store with multiplication by dedispersion kernel [no FFT batching]
//
__device__ void CB_convolve_and_store (void * dataOut, size_t offset, cufftComplex d, void * callerInfo, void *sharedPtr)
{
  // the dedispersion kernel complex float for this element of the FFT
  const cufftComplex k = ((cufftComplex *) callerInfo)[conv_params[0] + offset];
  ((cufftComplex*)dataOut)[offset] = cuCmulf (d, k);
}

__device__ void CB_convolve_and_store_batch (void * dataOut, size_t offset, cufftComplex d, void * callerInfo, void *sharedPtr)
{
  // the dedispersion kernel value for this element of the FFT
  const unsigned kernel_offset = conv_params[0] + (offset % conv_params[1]);
  const cufftComplex k = ((cufftComplex *) callerInfo)[kernel_offset];

  ((cufftComplex*)dataOut)[offset] = cuCmulf (d, k);
}
__device__ cufftCallbackStoreC d_store_fwd        = CB_convolve_and_store;
__device__ cufftCallbackStoreC d_store_fwd_batch = CB_convolve_and_store_batch;

/////////////////////////////////////////////////////////////////////////
//
// store with output filtering on
//
__device__ void CB_filtered_store (void * dataOut, size_t offset, cufftComplex d, void * callerInfo, void *sharedPtr)
{
  // if offset < nfilt_pos, discard
  if (offset < conv_params[2])
    return;

  // if offset > (npt - nfilt_neg), discard
  if (offset >= conv_params[3])
    return;

  ((cufftComplex*)dataOut)[offset - conv_params[2]] = d;
}

__device__ void CB_filtered_store_batch (void * dataOut, size_t offset, cufftComplex d, void * callerInfo, void *sharedPtr)
{
  const unsigned ibatch = offset / conv_params[1];
  const unsigned ipt = offset - (ibatch * conv_params[1]);

  // if ipt < nfilt_pos, discard
  if (ipt < conv_params[2])
    return;
  
  // if ipt > (npt - nfilt_neg), discard
  if (ipt >= conv_params[3])
    return;

  // substract the required offsets
  offset -= ((ibatch * conv_params[4]) + conv_params[2]);

  ((cufftComplex*)dataOut)[offset] = d;
}

__device__ cufftCallbackStoreC d_store_bwd       = CB_filtered_store;
__device__ cufftCallbackStoreC d_store_bwd_batch = CB_filtered_store_batch;

void setup_callbacks_ConvolutionCUDA (cufftHandle plan_fwd, cufftHandle plan_bwd, 
                                      cufftHandle plan_fwd_batched, cufftHandle plan_bwd_batched,
                                      cufftComplex * d_kernels, int nbatch, cudaStream_t stream)
{
  cudaError_t error;
  cufftResult_t result;

  cufftCallbackStoreC h_store_fwd;
  cufftCallbackStoreC h_store_bwd;
  cufftCallbackStoreC h_store_fwd_batch;
  cufftCallbackStoreC h_store_bwd_batch;

  error = cudaMemcpyFromSymbolAsync(&h_store_fwd, d_store_fwd, 
                                    sizeof(h_store_fwd), 0, 
                                    cudaMemcpyDeviceToHost, stream);
  if (error != cudaSuccess)
    throw Error (FailedCall, "CUDA::ConvolutionEngine::setup_callbacks",
                 "cudaMemcpyFromSymbolAsync failed for h_store_fwd");

  error = cudaMemcpyFromSymbolAsync(&h_store_bwd, d_store_bwd,
                                    sizeof(h_store_bwd), 0,
                                    cudaMemcpyDeviceToHost, stream);
  if (error != cudaSuccess)
    throw Error (FailedCall, "CUDA::ConvolutionEngine::setup_callbacks",
                 "cudaMemcpyFromSymbolAsync failed for h_store_bwd");

  error = cudaMemcpyFromSymbolAsync(&h_store_fwd_batch, d_store_fwd_batch,
                                    sizeof(h_store_fwd_batch), 0,
                                    cudaMemcpyDeviceToHost, stream);
  if (error != cudaSuccess)
    throw Error (FailedCall, "CUDA::ConvolutionEngine::setup_callbacks",
                 "cudaMemcpyFromSymbolAsync failed for h_store_fwd_batch");

  error = cudaMemcpyFromSymbolAsync(&h_store_bwd_batch, d_store_bwd_batch,
                                    sizeof(h_store_bwd_batch), 0,
                                    cudaMemcpyDeviceToHost, stream);
  if (error != cudaSuccess)
    throw Error (FailedCall, "CUDA::ConvolutionEngine::setup_callbacks",
                 "cudaMemcpyFromSymbolAsync failed for h_store_bwd_batch");

  result = cufftXtSetCallback (plan_fwd, (void **)&h_store_fwd,
                               CUFFT_CB_ST_COMPLEX, (void **)&d_kernels);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::setup_callbacks",
      "cufftXtSetCallback (plan_fwd, h_store_fwd)");

  result = cufftXtSetCallback (plan_bwd, (void **)&h_store_bwd,
                               CUFFT_CB_ST_COMPLEX, 0);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::setup_callbacks",
      "cufftXtSetCallback (plan_bwd, h_store_bwd)");

  if (nbatch > 0)
  {
    result = cufftXtSetCallback (plan_fwd_batched, (void **)&h_store_fwd_batch,
                                 CUFFT_CB_ST_COMPLEX, (void **)&d_kernels);
    if (result != CUFFT_SUCCESS)
      throw CUFFTError (result, "CUDA::ConvolutionEngine::setup_callbacks",
        "cufftXtSetCallback (plan_fwd_batched, h_store_fwd_batch)");

    result = cufftXtSetCallback (plan_bwd_batched, (void **)&h_store_bwd_batch,
                                 CUFFT_CB_ST_COMPLEX, 0);
    if (result != CUFFT_SUCCESS)
      throw CUFFTError (result, "CUDA::ConvolutionEngine::setup_callbacks",
        "cufftXtSetCallback (plan_bwd_batched, h_store_bwd_batch)");
  }
}

void setup_callbacks_conv_params (unsigned * h_ptr, unsigned h_size, cudaStream_t stream)
{
  cudaError_t error = cudaMemcpyToSymbolAsync (conv_params, (void *) h_ptr,
                                   h_size, 0,
                                   cudaMemcpyHostToDevice, stream);
  if (error != cudaSuccess)
  {
    throw Error (InvalidState, "CUDA::ConvolutionEngine::setup_kernel",
     "could not initialize convolution params in device memory");
  }

}

// 
//
//

// [0] first_ipt ( nfilt_pos )
// [1] last_ipt ( npt - nfilt_neg )
__device__ __constant__ unsigned conv_params_spectral[2];

/////////////////////////////////////////////////////////////////////////
//
// store with multiplication by dedispersion kernel
//
__device__ void CB_convolve_and_store_spectral (void * dataOut, size_t offset, cufftComplex d, void * callerInfo, void *sharedPtr)
{
  // the dedispersion kernel complex float for this element of the FFT
  const cufftComplex k = ((cufftComplex *) callerInfo)[offset];
  ((cufftComplex*)dataOut)[offset] = cuCmulf (d, k);
}
__device__ cufftCallbackStoreC d_store_fwd_spectral = CB_convolve_and_store_spectral;

/////////////////////////////////////////////////////////////////////////
//
// store with output filtering on
//
__device__ void CB_filtered_store_spectral (void * dataOut, size_t offset, cufftComplex d, void * callerInfo, void *sharedPtr)
{
  // if offset < nfilt_pos, discard
  if (offset < conv_params_spectral[0])
    return;

  // if offset > (npt - nfilt_neg), discard
  if (offset >= conv_params_spectral[1])
    return;

  ((cufftComplex*)dataOut)[offset - conv_params_spectral[0]] = d;
}
__device__ cufftCallbackStoreC d_store_bwd_spectral = CB_filtered_store_spectral;


void setup_callbacks_ConvolutionCUDASpectral (cufftHandle plan_fwd, cufftHandle plan_bwd, cufftComplex * d_kernels, cudaStream_t stream)
{
  cudaError_t error;
  cufftResult_t result;

  cufftCallbackStoreC h_store_fwd;
  cufftCallbackStoreC h_store_bwd;

  error = cudaMemcpyFromSymbolAsync(&h_store_fwd, d_store_fwd_spectral,
                                    sizeof(h_store_fwd), 0,
                                    cudaMemcpyDeviceToHost, stream);
  if (error != cudaSuccess)
    throw Error (FailedCall, "CUDA::ConvolutionEngineSpectral::setup_callbacks",
                 "cudaMemcpyFromSymbolAsync failed for h_store_fwd");

  error = cudaMemcpyFromSymbolAsync(&h_store_bwd, d_store_bwd_spectral,
                                    sizeof(h_store_bwd), 0,
                                    cudaMemcpyDeviceToHost, stream);
  if (error != cudaSuccess)
    throw Error (FailedCall, "CUDA::ConvolutionEngineSpectral::setup_callbacks",
                 "cudaMemcpyFromSymbolAsync failed for h_store_bwd");

  result = cufftXtSetCallback (plan_fwd, (void **)&h_store_fwd,
                               CUFFT_CB_ST_COMPLEX, (void **)&d_kernels);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngineSpectral::setup_callbacks",
      "cufftXtSetCallback (plan_fwd, h_store_fwd)");

  result = cufftXtSetCallback (plan_bwd, (void **)&h_store_bwd,
                               CUFFT_CB_ST_COMPLEX, 0);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngineSpectral::setup_callbacks",
      "cufftXtSetCallback (plan_bwd, h_store_bwd)");
}

void setup_callbacks_conv_params_spectral (unsigned * h_ptr, unsigned h_size, cudaStream_t stream)
{
  cudaError_t error = cudaMemcpyToSymbolAsync (conv_params_spectral, (void *) h_ptr,
                                   h_size, 0, cudaMemcpyHostToDevice, stream);
  if (error != cudaSuccess)
  {
    throw Error (InvalidState, "CUDA::ConvolutionEngineSpectral::setup_kernel",
     "could not initialize convolution params in device memory");
  }
}




#endif

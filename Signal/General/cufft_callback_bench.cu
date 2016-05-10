/***************************************************************************
 *
 *   Copyright (C) 2015 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include <cuda_runtime.h>
#include <cufft.h>
#include <cuda_fp16.h>
#include <cufftXt.h>

#include "CUFFTError.h"
#include "CommandLine.h"
#include "RealTimer.h"

#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <math.h>

using namespace std;

__global__ void k_unpack (cuFloatComplex * output, const __restrict__ char2 * input, const float scale)
{
  unsigned i = blockIdx.x*blockDim.x + threadIdx.x;

  char2 element = input[i];
  output[i] = make_cuComplex ((float) element.x/scale, (float) element.y/scale);
}

__global__ void k_multiply (float2* d_fft, const __restrict__ float2 * kernel, unsigned npart)
{
  const unsigned npt = blockDim.x * gridDim.x;
  unsigned i = blockIdx.x*blockDim.x + threadIdx.x;

  // load the kernel for this fine channel
  const float2 k = kernel[i];

  while (i < npt * npart)
  {
    const float2 d = d_fft[i];
    const float x = d.x * k.x - d.y * k.y;
    d_fft[i].y = d.x * k.y + d.y * k.x;
    d_fft[i].x = x;
    i += npt;
  }
}

/////////////////////////////////////////////////////////////////////////
//
// store with multiplication by dedispersion kernel
//
__device__ void CB_convolve_and_storeC (void * dataOut, size_t offset, cufftComplex d, void * callerInfo, void *sharedPtr)
{
  // the dedispersion kernel value for this element of the FFT
  const cufftComplex k = ((cufftComplex *) callerInfo)[offset];
  ((cufftComplex*)dataOut)[offset] = cuCmulf (d, k);
}

__device__ cufftCallbackStoreC d_store_fwd_C = CB_convolve_and_storeC;


/////////////////////////////////////////////////////////////////////////
//
// convert an 8bit number to 32 bit
//
__device__ cufftComplex cufft_callback_load_8bit(
    void *dataIn, 
    size_t offset, 
    void *callerInfo, 
    void *sharedPtr) 
{
  const __restrict__ char2 in = ((char2 *)dataIn)[offset];
  const float scale = 127.0f;
  return make_cuComplex ((float)in.x/scale, (float) in.y/scale);
  //char2 in = ((char2*)dataIn)[offset];
  //float2 out;
  //out.x = (float) in.x / scale;
  //out.y = (float) in.y / scale;

  //return out;
  //return make_cuComplex ((float) element.x, (float) element.y);
  //return make_cuComplex ((float) element.x/scale, (float) element.y/scale);
}
__device__ cufftCallbackLoadC d_load_8bit_fwd_C = cufft_callback_load_8bit;


/////////////////////////////////////////////////////////////////////////
//
// convert an 16bit number to 32 bit
//
__device__ cufftComplex cufft_callback_load_half2(
    void *dataIn,
    size_t offset,
    void *callerInfo,
    void *sharedPtr)
{
  half * ptr = (half*) dataIn + (2*offset);
  return make_cuComplex ( __half2float(ptr[0]), __half2float(ptr[1]));
}

__device__ cufftCallbackLoadC d_load_half2_fwd_C = cufft_callback_load_half2;


/////////////////////////////////////////////////////////////////////////
//
// store with output filtering on
//
__device__ void CB_filtered_store (void * dataOut, size_t offset, cufftComplex d, void * callerInfo, void *sharedPtr)
{
  unsigned nfilt_pos = ((unsigned *) callerInfo)[0];
  unsigned nsamp_filt = ((unsigned *) callerInfo)[1];

  offset -= nfilt_pos;
  if ((offset > 0) && (offset < nsamp_filt))
    ((cufftComplex*)dataOut)[offset] = d;
}

__device__ cufftCallbackStoreC d_store_bwd_C = CB_filtered_store;

class Speed : public Reference::Able
{
public:

  Speed ();

  // parse command line options
  void parseOptions (int argc, char** argv);

  // run the test
  void runTest ();

protected:

  int npt;
  int niter;
  unsigned gpu_id;
  bool cuda;
};


Speed::Speed ()
{
  gpu_id = 0;
  niter = 16;
  npt = 1024;
  cuda = false;
}

int main(int argc, char** argv) try
{
  Speed speed;
  speed.parseOptions (argc, argv);
  speed.runTest ();
  return 0;
}
catch (Error& error)
{
  cerr << error << endl;
  return -1;
}

void Speed::parseOptions (int argc, char** argv)
{
  CommandLine::Menu menu;
  CommandLine::Argument* arg;

  menu.set_help_header ("undersampling_speed - measure under sampling speed");
  menu.set_version ("undersampling_speed version 1.0");

  arg = menu.add (npt, 'n', "npt");
  arg->set_help ("number of points in each FFT");

#if HAVE_CUFFT
  arg = menu.add (gpu_id, 'd');
  arg->set_help ("GPU device ID");
#endif

  arg = menu.add (niter, 't', "ninter");
  arg->set_help ("number of iterations (batch/loops)");

#if HAVE_CUFFT
  arg = menu.add (cuda, "cuda");
  arg->set_help ("benchmark CUDA");
#endif

  menu.parse (argc, argv);
}

#if HAVE_CUFFT
void check_error_stream (const char*, cudaStream_t);
#endif

void Speed::runTest ()
{
#ifdef _DEBUG
  dsp::Operation::verbose = true;
  dsp::Observation::verbose = true;
#endif

  // assume complex FFTs
  const unsigned ndim = 2;
 
  cudaStream_t stream = 0;
  if (cuda)
  {
    cerr << "using GPU " << gpu_id << endl;
    cudaError_t err = cudaSetDevice(gpu_id); 
    if (err != cudaSuccess)
      throw Error (InvalidState, "undersampling_speed",
                   "cudaSetDevice failed: %s", cudaGetErrorString(err));

    err = cudaStreamCreate( &stream );
    if (err != cudaSuccess)
      throw Error (InvalidState, "undersampling_speed",
                   "cudaStreamCreate failed: %s", cudaGetErrorString(err));

  }

  const unsigned ndat = npt * niter;
  const unsigned raw_size = ndat * ndim * sizeof(int8_t);
  const unsigned half2_size = ndat * ndim * sizeof(half);
  const unsigned unpacked_size = ndat * ndim * sizeof(float);
  const unsigned kernel_size = npt * sizeof (cuFloatComplex);

  char2 * raw;
  half2 * input_h2;
  cufftComplex * input;
  cufftComplex * buffer;
  cufftComplex * output;
  cufftComplex * d_kernel;
  unsigned * d_offsets;
  cufftResult result;
  size_t work_size;

  cudaMalloc ((void **) &raw, raw_size);
  cudaMalloc ((void **) &input_h2, half2_size);
  cudaMalloc ((void **) &input, unpacked_size);
  cudaMalloc ((void **) &buffer, unpacked_size);
  cudaMalloc ((void **) &output, unpacked_size);
  cudaMalloc ((void **) &d_kernel, kernel_size);
  cudaMalloc ((void **) &d_offsets, 2 * sizeof(unsigned));

  cudaMemsetAsync ((void *) raw, 0, raw_size, stream);
  cudaMemsetAsync ((void *) input, 0, unpacked_size, stream);
  cudaMemsetAsync ((void *) input_h2, 0, half2_size, stream);
  cudaMemsetAsync ((void *) d_kernel, 0, kernel_size, stream);

  unsigned * h_offsets;
  cudaMallocHost((void **) &h_offsets, 2 * sizeof(unsigned));
  h_offsets[0] = (unsigned) (npt / 15);
  h_offsets[1] = (unsigned) (npt / 15);

  cudaMemcpyAsync ((void *) d_offsets, (void *) h_offsets, 2 * sizeof(unsigned), cudaMemcpyHostToDevice, stream);

  // all plans are using batched FFTs to ensure at least 1M points

  cufftHandle plan_batch;
  result = cufftCreate (&plan_batch);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "Speed::runTest", "cufftCreate(plan_batch)");

  int rank = 1;
  result = cufftMakePlanMany (plan_batch, rank, &npt, NULL, 0, 0, NULL, 0, 0, 
                              CUFFT_C2C, niter, &work_size);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "Speed::runTest", "cufftMakePlanMany (plan_batch)");

  result = cufftSetStream (plan_batch, stream);
  if (result != CUFFT_SUCCESS)
    CUFFTError (result, "Speed::runTest", "cufftSetStream (plan_batch)");


  cufftHandle plan_callback;
  result = cufftCreate (&plan_callback);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "Speed::runTest", "cufftCreate(plan_callback)");

  result = cufftMakePlanMany (plan_callback, rank, &npt, NULL, 0, 0, NULL, 0, 0,
                              CUFFT_C2C, niter, &work_size);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "Speed::runTest", "cufftMakePlanMany (plan_callback)");

  result = cufftSetStream (plan_callback, stream);
  if (result != CUFFT_SUCCESS)
    CUFFTError (result, "Speed::runTest", "cufftSetStream (plan_callback)");

  cufftHandle plan_half;
  result = cufftCreate (&plan_half);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "Speed::runTest", "cufftCreate(plan_half)");

  result = cufftMakePlanMany (plan_half, rank, &npt, NULL, 0, 0, NULL, 0, 0,
                              CUFFT_C2C, niter, &work_size);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "Speed::runTest", "cufftMakePlanMany (plan_half)");

  result = cufftSetStream (plan_half, stream);
  if (result != CUFFT_SUCCESS)
    CUFFTError (result, "Speed::runTest", "cufftSetStream (plan_half)");

  cufftHandle plan_bwd;
  result = cufftCreate (&plan_bwd);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "Speed::runTest", "cufftCreate(plan_bwd)");

  result = cufftMakePlanMany (plan_bwd, rank, &npt, NULL, 0, 0, NULL, 0, 0,
                              CUFFT_C2C, niter, &work_size);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "Speed::runTest", "cufftMakePlanMany (plan_bwd)");

  result = cufftSetStream (plan_bwd, stream);
  if (result != CUFFT_SUCCESS)
    CUFFTError (result, "Speed::runTest", "cufftSetStream (plan_bwd)");

  cufftHandle plan_bwd_cb;
  result = cufftCreate (&plan_bwd_cb);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "Speed::runTest", "cufftCreate(plan_bwd_cb)");
  
  result = cufftMakePlanMany (plan_bwd_cb, rank, &npt, NULL, 0, 0, NULL, 0, 0,
                              CUFFT_C2C, niter, &work_size);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "Speed::runTest", "cufftMakePlanMany (plan_bwd_cb)");

  result = cufftSetStream (plan_bwd_cb, stream);
  if (result != CUFFT_SUCCESS)
    CUFFTError (result, "Speed::runTest", "cufftSetStream (plan_bwd_cb)");




  RealTimer timer_batch;
  RealTimer timer_callback;
  RealTimer timer_half;
  RealTimer timer_;
 
  cufftCallbackLoadC  h_load_8bit_fwd_C;
  cufftCallbackLoadC  h_load_half2_fwd_C;
  cufftCallbackStoreC h_store_fwd_C;
  cufftCallbackStoreC h_store_bwd_C;
  cudaError_t error;

  error = cudaMemcpyFromSymbolAsync(&h_load_8bit_fwd_C,
                                    d_load_8bit_fwd_C,
                                    sizeof(h_load_8bit_fwd_C),
                                    0,
                                    cudaMemcpyDeviceToHost,
                                    stream);
  if (error != cudaSuccess)
    throw Error (FailedCall, "Speed::runTest",
                 "cudaMemcpyFromSymbolAsync failed for h_load_8bit_fwd_C");


  error = cudaMemcpyFromSymbolAsync(&h_load_half2_fwd_C,
                                    d_load_half2_fwd_C,
                                    sizeof(h_load_half2_fwd_C),
                                    0,
                                    cudaMemcpyDeviceToHost,
                                    stream);
  if (error != cudaSuccess)
    throw Error (FailedCall, "Speed::runTest",
                 "cudaMemcpyFromSymbolAsync failed for h_load_half2_fwd_C");

  error = cudaMemcpyFromSymbolAsync(&h_store_fwd_C,
                                    d_store_fwd_C,
                                    sizeof(h_store_fwd_C),
                                    0,
                                    cudaMemcpyDeviceToHost,
                                    stream);
  if (error != cudaSuccess)
    throw Error (FailedCall, "Speed::runTest",
                 "cudaMemcpyFromSymbolAsync failed for h_store_fwd_C");

  error = cudaMemcpyFromSymbolAsync(&h_store_bwd_C,
                                    d_store_bwd_C,
                                    sizeof(h_store_bwd_C),
                                    0,
                                    cudaMemcpyDeviceToHost,
                                    stream);
  if (error != cudaSuccess)
    throw Error (FailedCall, "Speed::runTest",
                 "cudaMemcpyFromSymbolAsync failed for h_store_bwd_C");

  result = cufftXtSetCallback (plan_callback,
                               (void **)&h_load_8bit_fwd_C,
                               CUFFT_CB_LD_COMPLEX,
                               0);
  if (result == CUFFT_LICENSE_ERROR)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::setup_callbacks", 
                      "CUFFT Callback invalid license");
  cerr << "result=" << result << endl;
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::setup_callbacks",
      "cufftXtSetCallback (plan_fwd, h_load_8bit_fwd_C)");

/*
  result = cufftXtSetCallback (plan_half,
                               (void **)&h_load_half2_fwd_C,
                               CUFFT_CB_LD_COMPLEX,
                               0);
  if (result == CUFFT_LICENSE_ERROR)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::setup_callbacks",
                      "CUFFT Callback invalid license");
  cerr << "result=" << result << endl;
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::setup_callbacks",
      "cufftXtSetCallback (plan_fwd, h_load_half2_fwd_C)");
*/

  result = cufftXtSetCallback (plan_callback,
                               (void **)&h_store_fwd_C,
                               CUFFT_CB_ST_COMPLEX,
                               (void **)&d_kernel);
  if (result == CUFFT_LICENSE_ERROR)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::setup_callbacks",
                      "CUFFT Callback invalid license");
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::setup_callbacks",
      "cufftXtSetCallback (plan_fwd, h_store_fwd_C)");

  result = cufftXtSetCallback (plan_bwd_cb,
                               (void **)&h_store_bwd_C,
                               CUFFT_CB_ST_COMPLEX,
                               (void **)&d_offsets);
  if (result == CUFFT_LICENSE_ERROR)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::setup_callbacks",
                      "CUFFT Callback invalid license");
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::setup_callbacks",
      "cufftXtSetCallback (plan_bwd_cb, h_store_bwd_C)");


  cudaStreamSynchronize (stream);
/*
  timer_half.start();

  result = cufftExecC2C (plan_half, (cufftComplex *) input_h2, output, CUFFT_FORWARD);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "Speed::runTest", "cufftExecC2C(plan_half)");
  cudaStreamSynchronize(stream);

  timer_half.stop();
*/

  timer_callback.start ();

  result = cufftExecC2C (plan_callback, (cuFloatComplex *) raw, buffer, CUFFT_FORWARD);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "Speed::runTest", "cufftExecC2C(plan_callback)");

  result = cufftExecC2C (plan_bwd_cb, output, buffer, CUFFT_INVERSE);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "Speed::runTest", "cufftExecC2C(plan_callback)");

  cudaStreamSynchronize(stream);

  timer_callback.stop ();
  double total_time, time_per_fft, time_us;

  total_time = timer_callback.get_elapsed();
  time_per_fft = total_time / niter;
  time_us = time_per_fft * 1e6;
  cerr << "CALLBACK: total_time=" << total_time << " time_per_fft=" << time_per_fft 
       << " time_us=" << time_us << endl;

  timer_batch.start ();

  unsigned nthreads = 1024;
  unsigned nblocks = ndat / nthreads;
  if (ndat % nthreads != 0)
    nblocks++;

  k_unpack<<<nblocks,nthreads,0,stream>>> (input, raw, 127.0f);

  result = cufftExecC2C (plan_batch, input, buffer, CUFFT_FORWARD);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "Speed::runTest", "cufftExecC2C(plan_batch)");

  nthreads = 1024;
  nblocks = npt / nthreads;
  if (npt % nthreads)
    nblocks++;

  k_multiply<<<nblocks,nthreads,0,stream>>> (buffer, d_kernel, niter);

  result = cufftExecC2C (plan_bwd, buffer, buffer, CUFFT_INVERSE);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "Speed::runTest", "cufftExecC2C(plan_callback)");

  cufftComplex * ou = output;
  cufftComplex * in = buffer;

  for (unsigned i=0; i<niter; i++)
  { 
    cudaMemcpyAsync ((void *) ou, (void *) in, npt * sizeof(cufftComplex), cudaMemcpyDeviceToDevice, stream);
    ou += npt;
    in += npt;
  }

  cudaStreamSynchronize(stream);

  timer_batch.stop ();

  total_time = timer_batch.get_elapsed();
  time_per_fft = total_time / niter;
  time_us = time_per_fft * 1e6;
  cerr << "BATCH: total_time=" << total_time << " time_per_fft=" << time_per_fft 
       << " time_us=" << time_us << endl;

  cufftDestroy(plan_callback);
  cufftDestroy(plan_batch);

  cudaFree(raw);
  cudaFree(input);
  cudaFree(output);
  cudaFree(d_kernel);
}

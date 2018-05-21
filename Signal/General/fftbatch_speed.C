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

#if CUDA_VERSION >= 8000
#define FP16
#endif

#ifdef FP16
#include <cufftXt.h>
#endif
#include "CUFFTError.h"

#include "CommandLine.h"
#include "RealTimer.h"

#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <math.h>

using namespace std;

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

  menu.set_help_header ("fftbatch_speed - measure FFT batch vs loop");
  menu.set_version ("fftbatch_speed version 1.0");

  arg = menu.add (npt, 'n', "npt");
  arg->set_help ("number of points in each FFT");

  arg = menu.add (gpu_id, 'd');
  arg->set_help ("GPU device ID");

  arg = menu.add (niter, 't', "ninter");
  arg->set_help ("number of iterations (batches or loops)");

  arg = menu.add (cuda, "cuda");
  arg->set_help ("benchmark CUDA");

  menu.parse (argc, argv);
}

void check_error_stream (const char*, cudaStream_t);

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
      throw Error (InvalidState, "fftbatch_speed",
                   "cudaSetDevice failed: %s", cudaGetErrorString(err));

    err = cudaStreamCreate( &stream );
    if (err != cudaSuccess)
      throw Error (InvalidState, "fftbatch_speed",
                   "cudaStreamCreate failed: %s", cudaGetErrorString(err));

  }

  unsigned ndat = npt * niter;
  unsigned nbytes = ndat * sizeof (cufftComplex);

  cufftComplex * input;
  cufftComplex * output;
  cufftResult result;
  size_t work_size;

  cudaMalloc ((void **) &input, nbytes);
  cudaMalloc ((void **) &output, nbytes);

  cudaMemsetAsync ((void *) input, 0, nbytes, stream);
  cudaMemsetAsync ((void *) output, 0, nbytes, stream);

  // setup loop and batch based FFT plans
  cufftHandle plan_loop;
  cufftHandle plan_batch;

  result = cufftCreate (&plan_loop);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "Speed::runTest", "cufftCreate(plan_loop)");

  result = cufftCreate (&plan_batch);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "Speed::runTest", "cufftCreate(plan_batch)");

  int rank = 1;

#ifdef FP16
  long long int n = (long long) npt;
  cudaDataType cuda_type = CUDA_C_16F;
  cudaDataType execution_type = CUDA_C_16F;

  result = cufftXtMakePlanMany (plan_loop, rank, &n,
                                NULL, 0, 0, cuda_type,
                                NULL, 0, 0, cuda_type,
                                1, &work_size,
                                execution_type);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "Speed::runTest", "cufftXtMakePlanMany (plan_loop)");

  result = cufftXtMakePlanMany (plan_batch, rank, &n, 
                                NULL, 0, 0, cuda_type,
	                        NULL, 0, 0, cuda_type, 
                                niter, &work_size,
  			        execution_type);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "Speed::runTest", "cufftXtMakePlanMany (plan_batch)");
#else
  result = cufftMakePlan1d (plan_loop, npt, CUFFT_C2C, 1, &work_size);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "Speed::runTest", "cufftMakePlan1D (plan_loop)");

  result = cufftMakePlanMany (plan_batch, rank, &npt, NULL, 0, 0, NULL, 0, 0, 
                              CUFFT_C2C, niter, &work_size);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "Speed::runTest", "cufftMakePlanMany (plan_batch)");
#endif

  result = cufftSetStream (plan_loop, stream);
  if (result != CUFFT_SUCCESS)
    CUFFTError (result, "Speed::runTest", "cufftSetStream (plan_loop)");

  result = cufftSetStream (plan_batch, stream);
  if (result != CUFFT_SUCCESS)
    CUFFTError (result, "Speed::runTest", "cufftSetStream (plan_batch)");

  RealTimer timer_loop;
  RealTimer timer_batch;

  cudaStreamSynchronize (stream);

  timer_loop.start ();

  for (unsigned i=0; i<niter; i++)
  {
#ifdef FP16
  result = cufftXtExec (plan_loop, input, output, CUFFT_FORWARD);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "Speed::runTest", "cufftXtExec(plan_loop)");
#else
    result = cufftExecC2C (plan_loop, input, output, CUFFT_FORWARD);
    if (result != CUFFT_SUCCESS)
      throw CUFFTError (result, "Speed::runTest", "cufftExecC2C(plan_loop)");
#endif
  }
  cudaStreamSynchronize(stream);

  timer_loop.stop ();

  double total_time, time_per_fft, time_us;

  total_time = timer_loop.get_elapsed();
  time_per_fft = total_time / niter;
  time_us = time_per_fft * 1e6;
  cerr << "LOOP: total_time=" << total_time << " time_per_fft=" << time_per_fft 
       << " time_us=" << time_us << endl;

  timer_batch.start ();

#ifdef FP16
  result = cufftXtExec (plan_batch, input, output, CUFFT_FORWARD);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "Speed::runTest", "cufftXtExec(plan_batch)");
#else
  result = cufftExecC2C (plan_batch, input, output, CUFFT_FORWARD);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "Speed::runTest", "cufftExecC2C(plan_batch)");
#endif
  cudaStreamSynchronize(stream);


  timer_batch.stop ();

  total_time = timer_batch.get_elapsed();
  time_per_fft = total_time / niter;
  time_us = time_per_fft * 1e6;
  cerr << "BATCH: total_time=" << total_time << " time_per_fft=" << time_per_fft 
       << " time_us=" << time_us << endl;

  cufftDestroy(plan_loop);
  cufftDestroy(plan_batch);
  cudaFree(input);
  cudaFree(output);

}


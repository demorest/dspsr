//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2015 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/ConvolutionCUDASpectral.h"
#include "CUFFTError.h"
#include "debug.h"

#if HAVE_CUFFT_CALLBACKS
#include "dsp/ConvolutionCUDACallbacks.h"
#include <cufftXt.h>
#endif

#include <iostream>
#include <cassert>

using namespace std;

void check_error_stream (const char*, cudaStream_t);

// ichan   == blockIdx.y
// ipt_bwd == blockIdx.x * blockDim.x + threadIdx.x
__global__ void k_multiply_conv_spectral (float2* d_fft, const __restrict__ float2 * kernel, unsigned npt_bwd)
{
  const unsigned idx = (blockIdx.y * npt_bwd) + (blockIdx.x * blockDim.x) + threadIdx.x;
  d_fft[idx] = cuCmulf(d_fft[idx], kernel[idx]);
}

// ichan == blockIdx.y
// ipt_bwd == blockIdx.x * blockDim.x + threadIdx.x
__global__ void k_ncopy_conv_spectral (float2* output_data, uint64_t ostride,
           const float2* input_data, uint64_t istride,
           unsigned nfilt_pos, unsigned nsamp_step)
{

  const unsigned idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (idx < nfilt_pos)
    return;

  uint64_t in_offset  = istride * blockIdx.y;
  uint64_t out_offset = ostride * blockIdx.y;

  unsigned isamp = idx;
  unsigned osamp = idx - nfilt_pos;

  if (osamp < nsamp_step)
    output_data[out_offset + osamp] = input_data[in_offset + isamp];
}

CUDA::ConvolutionEngineSpectral::ConvolutionEngineSpectral (cudaStream_t _stream)
{
  stream = _stream;

  // create plan handles
  cufftResult result;

  result = cufftCreate (&plan_fwd);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngineSpectral::ConvolutionEngineSpectral", 
                      "cufftCreate(plan_fwd)");

  result = cufftCreate (&plan_bwd);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngineSpectral::ConvolutionEngineSpectral", 
                      "cufftCreate(plan_bwd)");

  fft_configured = false;
  nchan = 0;
  npt_fwd = 0;
  npt_bwd = 0;

  work_area = 0;
  work_area_size = 0;

  buf = 0;
  d_kernels = 0;
}

CUDA::ConvolutionEngineSpectral::~ConvolutionEngineSpectral()
{
  cufftResult result;

  result = cufftDestroy (plan_fwd);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngineSpectral::~ConvolutionEngineSpectral",
                      "cufftDestroy(plan_fwd)");

  result = cufftDestroy (plan_bwd);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngineSpectral::~ConvolutionEngineSpectral",
                      "cufftDestroy(plan_bwd)");

  if (work_area)
  {
    cudaError_t error = cudaFree (work_area);
    if (error != cudaSuccess)
       throw Error (FailedCall, "CUDA::ConvolutionEngineSpectral::~ConvolutionEngineSpectral",
                    "cudaFree(%xu): %s", &work_area,
                     cudaGetErrorString (error));
  }

  if (buf)
  {
    cudaError_t error = cudaFree (buf);
    if (error != cudaSuccess)
       throw Error (FailedCall, "CUDA::ConvolutionEngineSpectral::~ConvolutionEngineSpectral",
                    "cudaFree(%xu): %s", &buf,
                     cudaGetErrorString (error));
  }
}

void CUDA::ConvolutionEngineSpectral::regenerate_plans()
{
  cufftResult result;
  result = cufftDestroy (plan_fwd);
  result = cufftCreate (&plan_fwd);

  result = cufftDestroy (plan_bwd);
  result = cufftCreate (&plan_bwd);
}

void CUDA::ConvolutionEngineSpectral::set_scratch (void * scratch)
{
  d_scratch = (cufftComplex *) scratch;
}

// prepare all relevant attributes for the engine
void CUDA::ConvolutionEngineSpectral::prepare (dsp::Convolution * convolution)
{
  const dsp::Response* response = convolution->get_response();

  nchan = response->get_nchan();
  npt_bwd = response->get_ndat();
  npt_fwd = convolution->get_minimum_samples();
  nsamp_overlap = convolution->get_minimum_samples_lost();
  nsamp_step = npt_fwd - nsamp_overlap;
  nfilt_pos = response->get_impulse_pos ();
  nfilt_neg = response->get_impulse_neg ();

  if (convolution->get_input()->get_state() == Signal::Nyquist)
    type_fwd = CUFFT_R2C;
  else
    type_fwd = CUFFT_C2C;

  // configure the dedispersion kernel
  setup_kernel (convolution->get_response());

  fft_configured = false;

  // initialize the kernel size configuration
  mp.init();
  mp.set_nelement (npt_bwd);
}

// setup the convolution kernel based on the reposnse
void CUDA::ConvolutionEngineSpectral::setup_kernel (const dsp::Response * response)
{
  unsigned nchan = response->get_nchan();
  unsigned ndat = response->get_ndat();
  unsigned ndim = response->get_ndim();

  assert (ndim == 2);
  assert (d_kernels == 0);

  if (dsp::Operation::verbose)
    cerr << "CUDA::ConvolutionEngineSpectral::setup_kernel response: "
         << "nchan=" << nchan << " ndat=" << ndat << " ndim=" << ndim << endl;

	// allocate memory for dedispersion kernel of all channels
	unsigned kernels_size = ndat * sizeof(cufftComplex) * nchan;
  cudaError_t error = cudaMalloc ((void**)&d_kernels, kernels_size);
  if (error != cudaSuccess)
  {
    throw Error (InvalidState, "CUDA::ConvolutionEngineSpectral::setup_kernel",
     "could not allocate device memory for dedispersion kernel");
  }

  // copy all kernels from host to device
  const float* kernel = response->get_datptr (0,0);

  if (dsp::Operation::verbose)
    cerr << "CUDA::ConvolutionEngineSpectral::setup_kernel cudaMemcpy stream=" 
         << stream << " size=" << kernels_size << endl;
  if (stream)
    error = cudaMemcpyAsync (d_kernels, kernel, kernels_size, cudaMemcpyHostToDevice, stream);
  else
    error = cudaMemcpy (d_kernels, kernel, kernels_size, cudaMemcpyHostToDevice);
  if (error != cudaSuccess)
  {
    throw Error (InvalidState, "CUDA::ConvolutionEngineSpectral::setup_kernel",
     "could not copy dedispersion kernel to device");
  }

#if HAVE_CUFFT_CALLBACKS
  error = cudaMallocHost ((void **) h_conv_params, sizeof(unsigned) * 2);
  if (error != cudaSuccess)
    throw Error (InvalidState, "CUDA::ConvolutionEngineSpectral::setup_kernel",
                 "could not allocate memory for h_conv_params");

  h_conv_params[0] = nfilt_pos;
  h_conv_params[1] = npt_bwd - nfilt_neg;
  setup_callbacks_conv_params_spectral (h_conv_params, sizeof (h_conv_params), stream);
#endif
}

// configure the batched FFT plans
void CUDA::ConvolutionEngineSpectral::setup_batched (const dsp::TimeSeries* input,
                                                     dsp::TimeSeries * output)
{
  if (dsp::Operation::verbose)
    cerr << "CUDA::ConvolutionEngineSpectral::setup_batched npt_fwd=" << npt_fwd 
         << " npt_bwd=" << npt_bwd << endl;

  nchan = input->get_nchan();
  npol  = input->get_npol();
  unsigned ndim = input->get_ndim();

#ifdef _DEBUG
  cerr << "CUDA::ConvolutionEngineSpectral::setup_batched nchan=" << nchan 
       << " npol=" << npol << " ndat=" << input->get_ndat() << endl;
#endif

  input_stride = (input->get_datptr (1, 0) - input->get_datptr (0, 0)) / ndim;
  output_stride = (output->get_datptr (1, 0) - output->get_datptr (0, 0) ) / ndim;

  int rank = 1; 
  int inembed[1];
  int onembed[1];
  int istride, ostride, idist, odist;
  cufftResult result;

  // now setup the forward batched plan
  size_t work_size_fwd, work_size_bwd;

  // complex layout plans for input
  inembed[0] = npt_fwd;
  onembed[0] = npt_bwd;

  istride = 1;
  ostride = 1;

  idist = (int) input_stride;
  odist = npt_bwd;

#ifdef _DEBUG
  cerr << "CUDA::ConvolutionEngineSpectral::setup_batched npt_fwd=" << npt_fwd 
       << " nbatch=" << nchan << endl;
  cerr << "CUDA::ConvolutionEngineSpectral::setup_batched input_stride=" 
       << input_stride << " output_stride=" << output_stride << endl;
#endif

  // setup forward fft
  result = cufftMakePlanMany (plan_fwd, rank, &npt_fwd, 
                              inembed, istride, idist,
                              onembed, ostride, odist,
                              type_fwd, nchan, &work_size_fwd);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngineSpectral::setup_batched", 
                      "cufftMakePlanMany (plan_fwd)");

  result = cufftSetStream (plan_fwd, stream);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngineSpectral::setup_batched",
          "cufftSetStream(plan_fwd)");

  // get a rough estimate on work buffer size
  work_size_fwd = 0;
  result = cufftEstimateMany(rank, &npt_fwd, 
                             inembed, istride, idist, 
                             onembed, ostride, odist, 
                             type_fwd, nchan, &work_size_fwd);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngineSpectral::setup_batched",
                      "cufftEstimateMany(plan_fwd)");

  istride = 1;
  ostride = 1;

#ifdef HAVE_CUFFT_CALLBACKS
  inembed[0] = npt_bwd;
  onembed[0] = nsamp_step;

  idist = npt_bwd;
  odist = (int) output_stride;
#else
  inembed[0] = npt_bwd;
  onembed[0] = npt_bwd;

  idist = npt_bwd;
  odist = npt_bwd;
#endif

  // the backward FFT is a has a simple layout (npt_bwd)
  DEBUG("CUDA::ConvolutionEngineSpectral::setup_batched cufftMakePlanMany (plan_bwd)");
  result = cufftMakePlanMany (plan_bwd, rank, &npt_bwd, 
                              inembed, istride, idist,
                              onembed, ostride, odist,
                              CUFFT_C2C, nchan, &work_size_bwd);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngineSpectral::setup_batched", 
                      "cufftMakePlanMany (plan_bwd)");

  result = cufftSetStream (plan_bwd, stream);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngineSpectral::setup_batched",
                      "cufftSetStream(plan_bwd)");

  DEBUG("CUDA::ConvolutionEngineSpectral::setup_batched bwd FFT plan set");

  work_size_bwd = 0;
  result = cufftEstimateMany(rank, &npt_bwd, 
                             inembed, istride, idist, 
                             onembed, ostride, odist, 
                             CUFFT_C2C, nchan, &work_size_bwd);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngineSpectral::setup_batched",
                      "cufftEstimateMany(plan_fwd)");
  
/*
  work_area_size = (work_size_fwd > work_size_bwd) ? work_size_fwd : work_size_bwd;
  auto_allocate = work_area_size > 0;

  DEBUG("CUDA::ConvolutionEngineSpectral::setup_batched cufftSetAutoAllocation(plan_fwd)");
  result = cufftSetAutoAllocation(plan_fwd, auto_allocate);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngineSpectral::setup_batched",
                      "cufftSetAutoAllocation(plan_bwd, %d)", 
                      auto_allocate);

  DEBUG("CUDA::ConvolutionEngineSpectral::setup_batched cufftSetAutoAllocation(plan_bwd)");
  result = cufftSetAutoAllocation(plan_bwd, auto_allocate);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngineSpectral::setup_batched",
                      "cufftSetAutoAllocation(plan_bwd, %d)", auto_allocate);

*/
  // free the space allocated for buf in setup_singular
  cudaError_t error;
  if (buf)
  {
    error = cudaFree (buf);
    if (error != cudaSuccess)
      throw Error (FailedCall, "CUDA::ConvolutionEngineSpectral::setup_batched",
                   "cudaFree(%x): %s", &buf, cudaGetErrorString (error));
  }

  size_t batched_buffer_size = npt_bwd * nchan * sizeof (cufftComplex);
  error = cudaMalloc ((void **) &buf, batched_buffer_size);
  if (error != cudaSuccess)
    throw Error (FailedCall, "CUDA::ConvolutionEngineSpectral::setup_batched",
                 "cudaMalloc(%x, %u): %s", &buf, batched_buffer_size,
                 cudaGetErrorString (error));

	// allocate device memory for dedispsersion kernel (1 channel)
/*
  if (work_area_size > 0)
  {
    if (work_area)
    {
      error = cudaFree (work_area);
      if (error != cudaSuccess)
         throw Error (FailedCall, "CUDA::ConvolutionEngineSpectral::setup",
                     "cudaFree(%xu): %s", &work_area,
                     cudaGetErrorString (error));
    }
    DEBUG("CUDA::ConvolutionEngineSpectral::setup cudaMalloc("<<work_area<<", "<<work_area_size<<")");
    error = cudaMalloc (&work_area, work_area_size);  
    if (error != cudaSuccess)
      throw Error (FailedCall, "CUDA::ConvolutionEngineSpectral::setup", 
                   "cudaMalloc(%x, %u): %s", &work_area, work_area_size,
                   cudaGetErrorString (error));
  }
  else
    work_area = 0;
*/
}

// Perform convolution choosing the optimal batched size or if ndat is not as
// was configured, then perform singular
void CUDA::ConvolutionEngineSpectral::perform (const dsp::TimeSeries* input, dsp::TimeSeries * output, unsigned npart)
{
  if (dsp::Operation::verbose)
    cerr << "CUDA::ConvolutionEngineSpectral::perform (" << npart << ")" << endl;

  if (npart == 0)
    return;

  uint64_t curr_istride = (input->get_datptr (1, 0) - input->get_datptr (0, 0)) / input->get_ndim();
  uint64_t curr_ostride = (output->get_datptr (1, 0) - output->get_datptr (0, 0)) / output->get_ndim();

  if (dsp::Operation::verbose)
  {
    cerr << "CUDA::ConvolutionEngineSpectral::perform istride prev=" << input_stride << " curr=" << curr_istride << " ndim=" << input->get_ndim() << endl;
    cerr << "CUDA::ConvolutionEngineSpectral::perform ostride prev=" << output_stride << " curr=" << curr_ostride << " ndim=" <<
output->get_ndim() << endl;
  }

  if (curr_istride != input_stride || curr_ostride != output_stride)
  {
    if (dsp::Operation::verbose)
      cerr << "CUDA::ConvolutionEngineSpectral::perform reconfiguring FFT batch sizes" << endl;
    fft_configured = false;
  }

  if (!fft_configured)
  {
    regenerate_plans ();
    setup_batched (input, output);
#if HAVE_CUFFT_CALLBACKS
    cerr << "CUDA::ConvolutionEngineSpectral::perform setup_callbacks_ConvolutionCUDASpectral()" << endl;
    setup_callbacks_ConvolutionCUDASpectral (plan_fwd, plan_bwd, d_kernels, stream);
#endif
    fft_configured = true;
  }

  if (type_fwd == CUFFT_C2C)
  {
    perform_complex (input, output, npart);
  }
  else
  {
    cerr << "CUDA::ConvolutionEngineSpectral::perform_real not implemented" << endl;
    //perform_real (input, output, npart);
  }
}

void CUDA::ConvolutionEngineSpectral::perform_complex (const dsp::TimeSeries* input, 
                                                       dsp::TimeSeries * output,
                                                       unsigned npart)
{
  const unsigned npol = input->get_npol();
  const unsigned nchan = input->get_nchan();
  const unsigned ndim = input->get_ndim();
  const uint64_t ipol_stride = input_stride / npol;
  const uint64_t opol_stride = output_stride / npol;

  cufftComplex * in;
  cufftComplex * out;
  cufftResult result;

	if (dsp::Operation::verbose)
  	cerr << "CUDA::ConvolutionEngineSpectral::perform_complex npart=" << npart 
				 << " nsamp_step=" << nsamp_step << endl;

#if !HAVE_CUFFT_CALLBACKS
  dim3 blocks = dim3 (npt_bwd / mp.get_nthread(), nchan);
  unsigned nthreads = mp.get_nthread();

  if (npt_bwd <= nthreads) 
  {
    blocks.x = 1;
    nthreads = npt_bwd;
  }
  else
  {
    if (npt_bwd % nthreads)
      blocks.x++;
  }
#endif

  cufftComplex * in_t  = (cufftComplex *) input->get_datptr (0, 0);
  cufftComplex * out_t = (cufftComplex *) output->get_datptr (0, 0);

  if (dsp::Operation::verbose)
    cerr << "CUDA::ConvolutionEngineSpectral::perform_complex in=" << in_t << " out=" << out_t << endl;

  for (unsigned ipart=0; ipart<npart; ipart++)
  {
    in  = in_t;
    out = out_t;

    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      // perform nchan batched forward FFTs for the current ipol and ipart
      result = cufftExecC2C (plan_fwd, in, buf, CUFFT_FORWARD);
      if (result != CUFFT_SUCCESS)
        throw CUFFTError (result, "CUDA::ConvolutionEngineSpectral::perform_complex",
                          "cufftExecC2C(plan_fwd)");

#if HAVE_CUFFT_CALLBACKS
      // perform the inverse batched FFT (out-of-place)
      result = cufftExecC2C (plan_bwd, buf, out, CUFFT_INVERSE);
      if (result != CUFFT_SUCCESS)
        throw CUFFTError (result, "CUDA::ConvolutionEngineSpectral::perform_complex",
                            "cufftExecC2C(plan_bwd)");

#else
      // multiply by the dedispersion kernel
      k_multiply_conv_spectral<<<blocks, nthreads, 0, stream>>> (buf, d_kernels, npt_bwd);

      // perform the inverse batched FFT (in-place)
      result = cufftExecC2C (plan_bwd, buf, buf, CUFFT_INVERSE);
      if (result != CUFFT_SUCCESS)
        throw CUFFTError (result, "CUDA::ConvolutionEngineSpectral::perform_complex",
                          "cufftExecC2C(plan_bwd)");

      // copy batches of output from input
      k_ncopy_conv_spectral<<<blocks, nthreads, 0, stream>>> (out, output_stride,
                                                              buf, npt_bwd,
                                                              nfilt_pos, nsamp_step);
#endif
      in  += ipol_stride;
      out += opol_stride;
    }

    in_t  += nsamp_step;
    out_t += nsamp_step;
  }

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error_stream( "CUDA::ConvolutionEngineSpectral::perform_complex", stream );
}

#if 0
void CUDA::ConvolutionEngineSpectral::perform_real(const dsp::TimeSeries* input,
                                           dsp::TimeSeries * output,
                                           unsigned npart)
{
  const unsigned npol = input->get_npol();
  const unsigned nchan = input->get_nchan();
  const unsigned ndim = input->get_ndim();

  cufftReal * in;
  cufftComplex * out;
  cufftResult result;

  const unsigned out_nsamp_step = nsamp_step / 2;

  const unsigned in_step_batch  = nsamp_step * nbatch;
  const unsigned out_step_batch = out_nsamp_step * nbatch;

  unsigned nbp = 0;
  if (nbatch > 0)
    nbp = npart / nbatch;

  dim3 blocks = dim3 (out_nsamp_step, nbatch, 0);
  if (out_nsamp_step % mp.get_nthread())
    blocks.x++;

  if (dsp::Operation::verbose)
    cerr << "CUDA::ConvolutionEngineSpectral::perform_real nsamp_step=" << nsamp_step
         << " npt_bwd=" << npt_bwd << endl;

  for (unsigned ichan=0; ichan<nchan; ichan++)
  {
    const unsigned k_offset = ichan * npt_bwd;

    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      in  = (cufftReal *) input->get_datptr (ichan, ipol);
      out = (cufftComplex *) output->get_datptr (ichan, ipol);

      // for each batched FFT
      for (unsigned i=0; i<nbp; i++)
      {
        // perform forward batched FFT
        result = cufftExecR2C (plan_fwd, in, buf);
        if (result != CUFFT_SUCCESS)
          throw CUFFTError (result, "CUDA::ConvolutionEngineSpectral::perform_real",
                            "cufftExecC2C(plan_fwd)");

        // multiply by the dedispersion kernel
        k_multiply_conv<<<mp.get_nblock(),mp.get_nthread(),0,stream>>> (buf,
                                                                   d_kernels + k_offset,
                                                                   nbatch);

        // perform the inverse batched FFT (in-place)
        result = cufftExecC2C (plan_bwd, buf, buf, CUFFT_INVERSE);
        if (result != CUFFT_SUCCESS)
          throw CUFFTError (result, "CUDA::ConvolutionEngineSpectral::perform_real",
                            "cufftExecC2C(plan_bwd)");

        // copy batches of output from input
        k_ncopy_conv<<<blocks,mp.get_nthread(),0,stream>>> (out, out_nsamp_step,
                                                       buf + nfilt_pos, npt_bwd,
                                                       out_step_batch);

        in  += in_step_batch;
        out += out_step_batch;
      }

      for (unsigned ipart=nbp*nbatch; ipart<npart; ipart++)
      {
        result = cufftExecR2C (plan_fwd, in, buf);
        if (result != CUFFT_SUCCESS)
          throw CUFFTError (result, "CUDA::ConvolutionEngineSpectral::perform_real",
                            "cufftExecC2C(plan_fwd)");

        // multiply by the dedispersion kernel
        k_multiply_conv<<<mp.get_nblock(),mp.get_nthread(),0,stream>>> (buf,
                                                                   d_kernels + k_offset,
                                                                   1);

        // perform the inverse batched FFT (in-place)
        result = cufftExecC2C (plan_bwd, buf, buf, CUFFT_INVERSE);
        if (result != CUFFT_SUCCESS)
          throw CUFFTError (result, "CUDA::ConvolutionEngineSpectral::perform",
                            "cufftExecC2C(plan_bwd)");

        // copy batches of output from input
        k_ncopy_conv<<<blocks.x,mp.get_nthread(),0,stream>>> (out, out_nsamp_step,
                                                         buf + nfilt_pos, npt_bwd,
                                                         out_step_batch);
        in  += nsamp_step;
        out += out_nsamp_step;
      }
    }
  }
  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error_stream( "CUDA::ConvolutionEngineSpectral::perform_real", stream );
}
#endif

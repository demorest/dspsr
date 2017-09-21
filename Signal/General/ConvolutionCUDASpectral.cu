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

#include <iostream>
#include <cassert>

using namespace std;

void check_error_stream (const char*, cudaStream_t);

// ichan  == blockIdx.y
// ipol   == blockIdx.z
// ipt_bwd == blockIdx.x * blockDim.x + threadIdx.x
__global__ void k_multiply_conv_spectral (float2* d_fft, const __restrict__ float2 * kernel, unsigned buf_stride, unsigned npt_bwd, unsigned npart)
{
  const unsigned ipt_bwd = blockIdx.x * blockDim.x + threadIdx.x;
  const float2 kern = kernel[blockIdx.y * npt_bwd + ipt_bwd];

  const unsigned ichanpol = blockIdx.y * gridDim.z + blockIdx.z;
  uint64_t idx = (ichanpol * buf_stride) + ipt_bwd;

  for (unsigned ipart=0; ipart<npart; ipart++)
  {
    d_fft[idx] = cuCmulf(d_fft[idx], kern);
    idx += npt_bwd;
  }
}

// ichan == blockIdx.y
// ipol  == blockIdx.z
// ipt_bwd == blockIdx.x * blockDim.x + threadIdx.x
__global__ void k_ncopy_conv_spectral (float2* output_data, uint64_t ostride,
           const float2* input_data, uint64_t istride, unsigned npt_bwd,
           unsigned nfilt_pos, unsigned nsamp_step, unsigned npart)
{
  const unsigned ipt_bwd = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (ipt_bwd < nfilt_pos)
    return;

  const unsigned osamp = ipt_bwd - nfilt_pos;
  if (osamp >= nsamp_step)
    return;

  const unsigned ichanpol = blockIdx.y * gridDim.z + blockIdx.z;
  uint64_t idx = istride * ichanpol + ipt_bwd;
  uint64_t odx = ostride * ichanpol + osamp;

  for (unsigned ipart=0; ipart<npart; ipart++)
  {
    output_data[odx] = input_data[idx];
    idx += npt_bwd;
    odx += nsamp_step;
  }
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
  cufftDestroy (plan_fwd);
  cufftCreate (&plan_fwd);
  cufftDestroy (plan_bwd);
  cufftCreate (&plan_bwd);
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

}

// configure the batched FFT plans
void CUDA::ConvolutionEngineSpectral::setup_batched (const dsp::TimeSeries* input,
                                                     dsp::TimeSeries * output, unsigned npart)
{
  if (dsp::Operation::verbose)
    cerr << "CUDA::ConvolutionEngineSpectral::setup_batched npt_fwd=" << npt_fwd 
         << " npt_bwd=" << npt_bwd << " npart=" << npart << endl;

  nchan = input->get_nchan();
  npol  = input->get_npol();
  unsigned ndim = input->get_ndim();

#ifdef _DEBUG
  cerr << "CUDA::ConvolutionEngineSpectral::setup_batched nchan=" << nchan 
       << " npol=" << npol << " ndat=" << input->get_ndat() << endl;
#endif

  input_stride = (input->get_datptr (1, 0) - input->get_datptr (0, 0)) / ndim;
  output_stride = (output->get_datptr (1, 0) - output->get_datptr (0, 0) ) / ndim;
  buf_stride = npt_bwd * npart * npol;

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
  odist = (int) buf_stride;

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

  result = cufftSetCompatibilityMode (plan_fwd, CUFFT_COMPATIBILITY_NATIVE);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngineSpectral::setup_batched",
          "cufftSetCompatibilityMode(plan_fwd)");

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

  inembed[0] = npt_bwd;
  onembed[0] = npt_bwd;

  idist = (int) buf_stride;
  odist = (int) buf_stride;

  // the backward FFT is a has a simple layout (npt_bwd)
  DEBUG("CUDA::ConvolutionEngineSpectral::setup_batched cufftMakePlanMany (plan_bwd)");
  result = cufftMakePlanMany (plan_bwd, rank, &npt_bwd, 
                              inembed, istride, idist,
                              onembed, ostride, odist,
                              CUFFT_C2C, nchan, &work_size_bwd);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngineSpectral::setup_batched", 
                      "cufftMakePlanMany (plan_bwd)");

  result = cufftSetCompatibilityMode(plan_bwd, CUFFT_COMPATIBILITY_NATIVE);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngineSpectral::setup_batched",
                      "cufftSetCompatibilityMode(plan_bwd)");

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
  
  // free the space allocated for buf in setup_singular
  cudaError_t error;
  size_t batched_buffer_size = npart * npt_bwd * nchan * npol * sizeof (cufftComplex);
  if (batched_buffer_size > buf_size)
  {
    error = cudaFree (buf);
    if (error != cudaSuccess)
      throw Error (FailedCall, "CUDA::ConvolutionEngineSpectral::setup_batched",
                   "cudaFree(%x): %s", &buf, cudaGetErrorString (error));

    error = cudaMalloc ((void **) &buf, batched_buffer_size);
    if (error != cudaSuccess)
      throw Error (FailedCall, "CUDA::ConvolutionEngineSpectral::setup_batched",
                   "cudaMalloc(%x, %u): %s", &buf, batched_buffer_size,
                   cudaGetErrorString (error));
    buf_size = batched_buffer_size;
  }
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
    cerr << "CUDA::ConvolutionEngineSpectral::perform ostride prev=" << output_stride << " curr=" << curr_ostride << " ndim=" << output->get_ndim() << endl;
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
    setup_batched (input, output, npart);
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
  const uint64_t bpol_stride = buf_stride / npol;

  cufftComplex * in_ptr;
  cufftComplex * buf_ptr;
  cufftResult result;

	if (dsp::Operation::verbose)
  	cerr << "CUDA::ConvolutionEngineSpectral::perform_complex npart=" << npart 
				 << " nsamp_step=" << nsamp_step << endl;

  dim3 blocks = dim3 (npt_bwd / mp.get_nthread(), nchan, npol);
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

  cufftComplex * in_t;
  cufftComplex * out_t;
  cufftComplex * buf_t;

  // forward FFT all the data for both polarisations (into FPT order)
  in_t  = (cufftComplex *) input->get_datptr (0, 0);
  buf_t = (cufftComplex *) buf;

  if (dsp::Operation::verbose)
    cerr << "CUDA::ConvolutionEngineSpectral::perform_complex in=" << in_t << " buf=" << buf_t << endl;

  for (unsigned ipart=0; ipart<npart; ipart++)
  {
    in_ptr  = in_t;
    buf_ptr = buf_t;

    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      result = cufftExecC2C (plan_fwd, in_ptr, buf_ptr, CUFFT_FORWARD);
      if (result != CUFFT_SUCCESS)
        throw CUFFTError (result, "CUDA::ConvolutionEngineSpectral::perform_complex",
                          "cufftExecC2C(plan_fwd)");
      in_ptr  += ipol_stride;
      buf_ptr += bpol_stride;
    }
    in_t  += nsamp_step;
    buf_t += npt_bwd;
  }

  // multiply by the dedispersion kernel across entire buf
  if (dsp::Operation::verbose)
    cerr << "CUDA::ConvolutionEngineSpectral::perform_complex k_multiply_conv_spectral bpol_stride=" << bpol_stride << endl;
  k_multiply_conv_spectral<<<blocks, nthreads, 0, stream>>> (buf, d_kernels, bpol_stride, npt_bwd, npart);

  buf_t = (cufftComplex *) buf;
  out_t = (cufftComplex *) output->get_datptr (0, 0);

  if (dsp::Operation::verbose)
    cerr << "CUDA::ConvolutionEngineSpectral::perform_complex buf=" << buf_t << " out=" << out_t << endl;


  // perform the inverse batched FFT (in-place)
  for (unsigned ipart=0; ipart<npart; ipart++)
  {
    buf_ptr = buf_t;

    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      result = cufftExecC2C (plan_bwd, buf_ptr, buf_ptr, CUFFT_INVERSE);
      if (result != CUFFT_SUCCESS)
      throw CUFFTError (result, "CUDA::ConvolutionEngineSpectral::perform_complex",
                        "cufftExecC2C(plan_bwd)");

      buf_ptr += bpol_stride;
    }
    buf_t += npt_bwd;
  }

  // copy back from buf to output
  out_t = (cufftComplex *) output->get_datptr (0, 0);
  k_ncopy_conv_spectral<<<blocks, nthreads, 0, stream>>> (out_t, opol_stride, buf, bpol_stride,
                                                          npt_bwd, nfilt_pos, nsamp_step, npart);
  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error_stream( "CUDA::ConvolutionEngineSpectral::perform_complex", stream );
}

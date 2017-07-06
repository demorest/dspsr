//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2015 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/ConvolutionCUDA.h"
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

__global__ void k_multiply_conv (float2* d_fft, const __restrict__ float2 * kernel, unsigned npart)
{
  const unsigned npt = blockDim.x * gridDim.x;
  unsigned i = blockIdx.x*blockDim.x + threadIdx.x;

  // load the kernel for this fine channel
  const float2 k = kernel[i];

  while (i < npt * npart)
  {
    d_fft[i] = cuCmulf(d_fft[i], k);
    i += npt;
  }
}

__global__ void k_ncopy_conv (float2* output_data, unsigned output_stride,
           const float2* input_data, unsigned input_stride,
           unsigned to_copy)
{
  // shift the input forward FFT by the required number of batches
  input_data += blockIdx.y * input_stride;

  // shift in output forward 
  output_data += blockIdx.y * output_stride;

  unsigned index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < to_copy)
    output_data[index] = input_data[index];
}


#if HAVE_CUFFT_CALLBACKS
/*
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
__device__ cufftCallbackStoreC d_store_fwd 			 = CB_convolve_and_store;
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
*/
#endif

CUDA::ConvolutionEngine::ConvolutionEngine (cudaStream_t _stream)
{
  stream = _stream;

  // create plan handles
  cufftResult result;

  result = cufftCreate (&plan_fwd);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::ConvolutionEngine", 
                      "cufftCreate(plan_fwd)");

  result = cufftCreate (&plan_fwd_batched);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::ConvolutionEngine", 
                      "cufftCreate(plan_fwd_batched)");

  result = cufftCreate (&plan_bwd);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::ConvolutionEngine", 
                      "cufftCreate(plan_bwd)");

  result = cufftCreate (&plan_bwd_batched);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::ConvolutionEngine", 
                      "cufftCreate(plan_bwd_batched)");

  nbatch = 0;
  npt_fwd = 0;
  npt_bwd = 0;

  work_area = 0;
  work_area_size = 0;

  buf = 0;
  d_kernels = 0;
}

CUDA::ConvolutionEngine::~ConvolutionEngine()
{
  cufftResult result;

  result = cufftDestroy (plan_fwd);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::~ConvolutionEngine",
                      "cufftDestroy(plan_fwd)");

  result = cufftDestroy (plan_fwd_batched);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::~ConvolutionEngine", 
                      "cufftDestroy(plan_fwd)");

  result = cufftDestroy (plan_bwd);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::~ConvolutionEngine",
                      "cufftDestroy(plan_bwd)");

  result = cufftDestroy (plan_bwd_batched);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::~ConvolutionEngine",
                      "cufftDestroy(plan_bwd_batched)");

  if (work_area)
  {
    cudaError_t error = cudaFree (work_area);
    if (error != cudaSuccess)
       throw Error (FailedCall, "CUDA::ConvolutionEngine::~ConvolutionEngine",
                    "cudaFree(%xu): %s", &work_area,
                     cudaGetErrorString (error));
  }
}

void CUDA::ConvolutionEngine::set_scratch (void * scratch)
{
  d_scratch = (cufftComplex *) scratch;
}

// prepare all relevant attributes for the engine
void CUDA::ConvolutionEngine::prepare (dsp::Convolution * convolution)
{
  const dsp::Response* response = convolution->get_response();

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

  // configure the singular FFT
  setup_singular ();

  // it is only more efficient to batch about to about 1M points 
  // at least on the TitanX, so lets choose the number of batches 
  // based on that
  unsigned npart = 1048576 / npt_fwd;

  if (npart > 1)
    setup_batched (npart);
  else
    nbatch = 0;

#if HAVE_CUFFT_CALLBACKS
  setup_callbacks_ConvolutionCUDA (plan_fwd, plan_bwd, plan_fwd_batched, plan_bwd_batched, d_kernels, nbatch, stream);
#endif

  // initialize the kernel size configuration
  mp.init();
  mp.set_nelement (npt_bwd);
}

// setup the convolution kernel based on the reposnse
void CUDA::ConvolutionEngine::setup_kernel (const dsp::Response * response)
{
  unsigned nchan = response->get_nchan();
  unsigned ndat = response->get_ndat();
  unsigned ndim = response->get_ndim();

  assert (ndim == 2);
  assert (d_kernels == 0);

	// allocate memory for dedispersion kernel of all channels
	unsigned kernels_size = ndat * sizeof(cufftComplex) * nchan;
  cudaError_t error = cudaMalloc ((void**)&d_kernels, kernels_size);
  if (error != cudaSuccess)
  {
    throw Error (InvalidState, "CUDA::ConvolutionEngine::setup_kernel",
     "could not allocate device memory for dedispersion kernel");
  }

  // copy all kernels from host to device
  const float* kernel = response->get_datptr (0,0);

  cerr << "CUDA::ConvolutionEngine::setup_kernel cudaMemcpy stream=" << stream 
       << " size=" << kernels_size << endl;
  if (stream)
    error = cudaMemcpyAsync (d_kernels, kernel, kernels_size, cudaMemcpyHostToDevice, stream);
  else
    error = cudaMemcpy (d_kernels, kernel, kernels_size, cudaMemcpyHostToDevice);
  if (error != cudaSuccess)
  {
    throw Error (InvalidState, "CUDA::ConvolutionEngine::setup_kernel",
     "could not copy dedispersion kernel to device");
  }

#if HAVE_CUFFT_CALLBACKS
  error = cudaMallocHost ((void **) h_conv_params, sizeof(unsigned) * 5);
  if (error != cudaSuccess)
    throw Error (InvalidState, "CUDA::ConvolutionEngine::setup_kernel",
                 "could not allocate memory for h_conv_params");

  h_conv_params[0] = 0;
  h_conv_params[1] = npt_bwd;
  h_conv_params[2] = nfilt_pos;
  h_conv_params[3] = npt_bwd - nfilt_neg;
  h_conv_params[4] = nfilt_pos + nfilt_neg;

  setup_callbacks_conv_params (h_conv_params, sizeof(h_conv_params), stream);

#endif
}

void CUDA::ConvolutionEngine::setup_singular ()
{
  if (dsp::Operation::verbose)
    cerr << "CUDA::ConvolutionEngine::setup_singular fwd=" << npt_fwd 
         << " bwd=" << npt_bwd << endl;

  // setup forward plan
  cufftResult result = cufftPlan1d (&plan_fwd, npt_fwd, type_fwd, 1);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::setup_singular",
                      "cufftPlan1d(plan_fwd)");

  result = cufftSetStream (plan_fwd, stream);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::setup_singular",
          "cufftSetStream(plan_fwd)");

  // setup backward plan
  result = cufftPlan1d (&plan_bwd, npt_bwd, CUFFT_C2C, 1);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::setup_singular",
                      "cufftPlan1d(plan_bwd)");

  result = cufftSetStream (plan_bwd, stream);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::setup_singular",
                      "cufftSetStream(plan_bwd)");

  size_t buffer_size = npt_bwd * sizeof (cufftComplex);
  cudaError_t error = cudaMalloc ((void **) &buf, buffer_size);
  if (error != cudaSuccess)
    throw Error (FailedCall, "CUDA::ConvolutionEngine::setup_singular",
                 "cudaMalloc(%x, %u): %s", &buf, buffer_size,
                 cudaGetErrorString (error));
}


// configure the singular and batched FFT plans
void CUDA::ConvolutionEngine::setup_batched (unsigned _nbatch)
{
  if (dsp::Operation::verbose)
    cerr << "CUDA::ConvolutionEngine::setup_batched npt_fwd=" << npt_fwd 
         << " npt_bwd=" << npt_bwd << " nbatch=" << _nbatch << endl;

  nbatch = _nbatch;

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

  // the fordward FFT only moves forward a shorter amount
  idist = nsamp_step;
  odist = npt_bwd;

  // setup forward fft
  result = cufftMakePlanMany (plan_fwd_batched, rank, &npt_fwd, 
                              inembed, istride, idist,
                              onembed, ostride, odist,
                              type_fwd, nbatch, &work_size_fwd);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::setup_batched", 
                      "cufftMakePlanMany (plan_fwd_batched)");

  result = cufftSetStream (plan_fwd_batched, stream);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::setup_batched",
          "cufftSetStream(plan_fwd_batched)");

  // get a rough estimate on work buffer size
  work_size_fwd = 0;
  result = cufftEstimateMany(rank, &npt_fwd, 
                             inembed, istride, idist, 
                             onembed, ostride, odist, 
                             type_fwd, nbatch, &work_size_fwd);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::setup_batched",
                      "cufftEstimateMany(plan_fwd)");

  // complex layout plans for input
  inembed[0] = npt_bwd;
  onembed[0] = nsamp_step;

  istride = 1;
  ostride = 1;

  // the fordward FFT only moves forward a shorter amount
  idist = npt_bwd;
  odist = nsamp_step;

  // the backward FFT is a has a simple layout (npt_bwd)
  DEBUG("CUDA::ConvolutionEngine::setup_batched cufftMakePlanMany (plan_bwd_batched)");
  result = cufftMakePlanMany (plan_bwd_batched, rank, &npt_bwd, 
                              inembed, istride, idist,
                              onembed, ostride, odist,
                              CUFFT_C2C, nbatch, &work_size_bwd);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::setup_batched", 
                      "cufftMakePlanMany (plan_bwd_batched)");

  result = cufftSetStream (plan_bwd_batched, stream);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::setup_batched",
                      "cufftSetStream(plan_bwd_batched)");

  DEBUG("CUDA::ConvolutionEngine::setup_batched bwd FFT plan set");

  work_size_bwd = 0;
  result = cufftEstimateMany(rank, &npt_bwd, 
                             inembed, istride, idist, 
                             onembed, ostride, odist, 
                             CUFFT_C2C, nbatch, &work_size_bwd);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::setup_batched",
                      "cufftEstimateMany(plan_fwd)");
  
  work_area_size = (work_size_fwd > work_size_bwd) ? work_size_fwd : work_size_bwd;
  auto_allocate = work_area_size > 0;

  DEBUG("CUDA::ConvolutionEngine::setup_batched cufftSetAutoAllocation(plan_fwd)");
  result = cufftSetAutoAllocation(plan_fwd_batched, auto_allocate);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::setup_batched",
                      "cufftSetAutoAllocation(plan_bwd_batched, %d)", 
                      auto_allocate);

  DEBUG("CUDA::ConvolutionEngine::setup_batched cufftSetAutoAllocation(plan_bwd_batched)");
  result = cufftSetAutoAllocation(plan_bwd_batched, auto_allocate);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::setup_batched",
                      "cufftSetAutoAllocation(plan_bwd_batched, %d)", auto_allocate);

  // free the space allocated for buf in setup_singular
  cudaError_t error = cudaFree (buf);
  if (error != cudaSuccess)
    throw Error (FailedCall, "CUDA::ConvolutionEngine::setup_batched",
                 "cudaFree(%x): %s", &buf, cudaGetErrorString (error));

  size_t batched_buffer_size = npt_bwd * nbatch * sizeof (cufftComplex);
  error = cudaMalloc ((void **) &buf, batched_buffer_size);
  if (error != cudaSuccess)
    throw Error (FailedCall, "CUDA::ConvolutionEngine::setup_batched",
                 "cudaMalloc(%x, %u): %s", &buf, batched_buffer_size,
                 cudaGetErrorString (error));

	// allocate device memory for dedispsersion kernel (1 channel)

  if (work_area_size > 0)
  {
    if (work_area)
    {
      error = cudaFree (work_area);
      if (error != cudaSuccess)
         throw Error (FailedCall, "CUDA::ConvolutionEngine::setup",
                     "cudaFree(%xu): %s", &work_area,
                     cudaGetErrorString (error));
    }
    DEBUG("CUDA::ConvolutionEngine::setup cudaMalloc("<<work_area<<", "<<work_area_size<<")");
    error = cudaMalloc (&work_area, work_area_size);  
    if (error != cudaSuccess)
      throw Error (FailedCall, "CUDA::ConvolutionEngine::setup", 
                   "cudaMalloc(%x, %u): %s", &work_area, work_area_size,
                   cudaGetErrorString (error));
  }
  else
    work_area = 0;
}

#if HAVE_CUFFT_CALLBACKS
/*
void CUDA::ConvolutionEngine::setup_callbacks ()
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
*/
#endif


// Perform convolution choosing the optimal batched size or if ndat is not as
// was configured, then perform singular
void CUDA::ConvolutionEngine::perform (const dsp::TimeSeries* input, dsp::TimeSeries * output, unsigned npart)
{
  if (dsp::Operation::verbose)
    cerr << "CUDA::ConvolutionEngine::perform (" << npart << ")" << endl;

  if (npart == 0)
    return;

  if (type_fwd == CUFFT_C2C)
    perform_complex (input, output, npart);
  else
    perform_real (input, output, npart);

}

void CUDA::ConvolutionEngine::perform_complex (const dsp::TimeSeries* input, 
                                               dsp::TimeSeries * output,
                                               unsigned npart)
{
  const unsigned npol = input->get_npol();
  const unsigned nchan = input->get_nchan();
  const unsigned ndim = input->get_ndim();

  cufftComplex * in;
  cufftComplex * out;
  cufftResult result;

  const unsigned in_step_batch  = nsamp_step * nbatch;
  const unsigned out_step_batch = nsamp_step * nbatch;

  unsigned nbp = 0;
  if (nbatch > 0)
    nbp = npart / nbatch;

	if (dsp::Operation::verbose)
  	cerr << "CUDA::ConvolutionEngine::perform_complex npart=" << npart 
         << " nbatch=" << nbatch 
				 << " npb=" << nbp << " nsamp_step=" << nsamp_step << endl;

#if !HAVE_CUFFT_CALLBACKS
  dim3 blocks = dim3 (nsamp_step, nbatch, 0);
  if (nsamp_step % mp.get_nthread())
    blocks.x++;
#endif

  for (unsigned ichan=0; ichan<nchan; ichan++)
  {

#if HAVE_CUFFT_CALLBACKS
    // determine convolution kernel offset
    h_conv_params[0] = ichan * npt_bwd;

    setup_callbacks_conv_params (h_conv_params, sizeof(unsigned), stream);

/*
		// update the channel offset in constant memory
		cudaError_t error = cudaMemcpyToSymbolAsync (conv_params, (void *) &h_conv_params, 
                          sizeof(unsigned), 0, cudaMemcpyHostToDevice, stream);
    if (error != cudaSuccess)
      throw Error (InvalidState, "CUDA::ConvolutionEngine::setup_kernel",
                   "could not update conv_params in device memory");
*/
#else
    const unsigned k_offset = ichan * npt_bwd;
#endif

    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      in  = (cufftComplex *) input->get_datptr (ichan, ipol);
      out = (cufftComplex *) output->get_datptr (ichan, ipol);

      // for each batched FFT
      for (unsigned i=0; i<nbp; i++)
      {
        // perform forward batched FFT
        result = cufftExecC2C (plan_fwd_batched, in, buf, CUFFT_FORWARD);
        if (result != CUFFT_SUCCESS)
          throw CUFFTError (result, "CUDA::ConvolutionEngine::perform_complex",
                            "cufftExecC2C(plan_fwd_batched)");

#if HAVE_CUFFT_CALLBACKS
        // perform the inverse batched FFT (out-of-place)
        result = cufftExecC2C (plan_bwd_batched, buf, out, CUFFT_INVERSE);
        if (result != CUFFT_SUCCESS)
          throw CUFFTError (result, "CUDA::ConvolutionEngine::perform_complex",
                            "cufftExecC2C(plan_bwd_batched)");

#else
        // multiply by the dedispersion kernel
        k_multiply_conv<<<mp.get_nblock(),mp.get_nthread(),0,stream>>> (buf,
                                                                        d_kernels + k_offset,
                                                                        nbatch);

        // perform the inverse batched FFT (in-place)
        result = cufftExecC2C (plan_bwd_batched, buf, buf, CUFFT_INVERSE);
        if (result != CUFFT_SUCCESS)
          throw CUFFTError (result, "CUDA::ConvolutionEngine::perform_complex",
                            "cufftExecC2C(plan_bwd_batched)");

        // copy batches of output from input
        k_ncopy_conv<<<blocks,mp.get_nthread(),0,stream>>> (out, nsamp_step,
                                                       buf + nfilt_pos, npt_bwd,
                                                       out_step_batch);
#endif

        out += out_step_batch;
        in  += in_step_batch;
      }

      for (unsigned ipart=nbp*nbatch; ipart<npart; ipart++)
      {
        result = cufftExecC2C (plan_fwd, in, buf, CUFFT_FORWARD);
        if (result != CUFFT_SUCCESS)
          throw CUFFTError (result, "CUDA::ConvolutionEngine::perform_complex",
                            "cufftExecC2C(plan_fwd)");

#if HAVE_CUFFT_CALLBACKS
        result = cufftExecC2C (plan_bwd, buf, out, CUFFT_INVERSE);
        if (result != CUFFT_SUCCESS)
          throw CUFFTError (result, "CUDA::ConvolutionEngine::perform_complex",
                            "cufftExecC2C(plan_bwd)");
#else
            // multiply by the dedispersion kernel
        k_multiply_conv<<<mp.get_nblock(),mp.get_nthread(),0,stream>>> (buf,
                                                                   d_kernels + k_offset,
                                                                   1);

        // perform the inverse batched FFT (in-place)
        result = cufftExecC2C (plan_bwd, buf, buf, CUFFT_INVERSE);
        if (result != CUFFT_SUCCESS)
          throw CUFFTError (result, "CUDA::ConvolutionEngine::perform",
                            "cufftExecC2C(plan_bwd_batched)");

        // copy batches of output from input
        k_ncopy_conv<<<blocks.x,mp.get_nthread(),0,stream>>> (out, nsamp_step,
                                                         buf + nfilt_pos, npt_bwd,
                                                         nsamp_step);
#endif

        in  += nsamp_step;
        out += nsamp_step;
      }
    }
  }

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error_stream( "CUDA::ConvolutionEngine::perform_complex", stream );
}

void CUDA::ConvolutionEngine::perform_real(const dsp::TimeSeries* input,
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
    cerr << "CUDA::ConvolutionEngine::perform_real nsamp_step=" << nsamp_step
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
        result = cufftExecR2C (plan_fwd_batched, in, buf);
        if (result != CUFFT_SUCCESS)
          throw CUFFTError (result, "CUDA::ConvolutionEngine::perform_real",
                            "cufftExecC2C(plan_fwd_batched)");

        // multiply by the dedispersion kernel
        k_multiply_conv<<<mp.get_nblock(),mp.get_nthread(),0,stream>>> (buf,
                                                                   d_kernels + k_offset,
                                                                   nbatch);

        // perform the inverse batched FFT (in-place)
        result = cufftExecC2C (plan_bwd_batched, buf, buf, CUFFT_INVERSE);
        if (result != CUFFT_SUCCESS)
          throw CUFFTError (result, "CUDA::ConvolutionEngine::perform_real",
                            "cufftExecC2C(plan_bwd_batched)");

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
          throw CUFFTError (result, "CUDA::ConvolutionEngine::perform_real",
                            "cufftExecC2C(plan_fwd)");

        // multiply by the dedispersion kernel
        k_multiply_conv<<<mp.get_nblock(),mp.get_nthread(),0,stream>>> (buf,
                                                                   d_kernels + k_offset,
                                                                   1);

        // perform the inverse batched FFT (in-place)
        result = cufftExecC2C (plan_bwd, buf, buf, CUFFT_INVERSE);
        if (result != CUFFT_SUCCESS)
          throw CUFFTError (result, "CUDA::ConvolutionEngine::perform",
                            "cufftExecC2C(plan_bwd_batched)");

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
    check_error_stream( "CUDA::ConvolutionEngine::perform_real", stream );
}

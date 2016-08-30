//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

//#define _DEBUG 1

#include "dsp/FilterbankCUDA.h"
#include "CUFFTError.h"
#include "debug.h"

#include <cuda_runtime.h>

#include <iostream>
#include <assert.h>

void check_error_stream (const char*, cudaStream_t);

#ifdef _DEBUG
#define CHECK_ERROR(x,y) check_error_stream(x,y)
#else
#define CHECK_ERROR(x,y)
#endif


__global__ void k_multiply (float2* d_fft, float2* kernel)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  float x = d_fft[i].x * kernel[i].x - d_fft[i].y * kernel[i].y;
  d_fft[i].y = d_fft[i].x * kernel[i].y + d_fft[i].y * kernel[i].x;
  d_fft[i].x = x;
}

__global__ void k_ncopy (float2* output_data, unsigned output_stride,
           const float2* input_data, unsigned input_stride,
           unsigned to_copy)
{
  output_data += blockIdx.y * output_stride;
  input_data += blockIdx.y * input_stride;

  unsigned index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < to_copy)
    output_data[index] = input_data[index];
}


using namespace std;

CUDA::FilterbankEngine::FilterbankEngine (cudaStream_t _stream)
{
  real_to_complex = false;

  d_fft = d_kernel = 0;

  stream = _stream;

  nfilt_pos = 0;
  plan_fwd = 0;
  plan_bwd = 0;
  verbose = false;
}

CUDA::FilterbankEngine::~FilterbankEngine ()
{
}

void CUDA::FilterbankEngine::setup (dsp::Filterbank* filterbank)
{
  // the CUDA engine does not maintain/compute the passband
  filterbank->set_passband (NULL);
  
  freq_res = filterbank->get_freq_res ();
  nchan_subband = filterbank->get_nchan_subband();

  real_to_complex = (filterbank->get_input()->get_state() == Signal::Nyquist);

  DEBUG("CUDA::FilterbankEngine::setup nchan_subband=" << nchan_subband \
        << " freq_res=" << freq_res);

  DEBUG("CUDA::FilterbankEngine::setup scratch=" << scratch);

  cufftResult result;
  if (real_to_complex)
  {
    DEBUG("CUDA::FilterbankEngine::setup plan size=" << freq_res*nchan_subband*2);
    result = cufftPlan1d (&plan_fwd, freq_res*nchan_subband*2, CUFFT_R2C, 1);
    if (result != CUFFT_SUCCESS)
      throw CUFFTError (result, "CUDA::FilterbankEngine::setup",
			"cufftPlan1d(plan_fwd, CUFFT_R2C)");
  }
  else
  {
    DEBUG("CUDA::FilterbankEngine::setup plan size=" << freq_res*nchan_subband);
    result = cufftPlan1d (&plan_fwd, freq_res*nchan_subband, CUFFT_C2C, 1);
    if (result != CUFFT_SUCCESS)
      throw CUFFTError (result, "CUDA::FilterbankEngine::setup",
			"cufftPlan1d(plan_fwd, CUFFT_C2C)");
  }

  result = cufftSetCompatibilityMode(plan_fwd, CUFFT_COMPATIBILITY_NATIVE);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::FilterbankEngine::setup",
		      "cufftSetCompatibilityMode(plan_fwd)");

  DEBUG("CUDA::FilterbankEngine::setup setting stream=" << stream);
  result = cufftSetStream (plan_fwd, stream);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::FilterbankEngine::setup", 
		      "cufftSetStream(plan_fwd)");

  DEBUG("CUDA::FilterbankEngine::setup fwd FFT plan set");

  if (freq_res > 1)
  {
    result = cufftPlan1d (&plan_bwd, freq_res, CUFFT_C2C, nchan_subband);
    if (result != CUFFT_SUCCESS)
      throw CUFFTError (result, "CUDA::FilterbankEngine::setup", 
			"cufftPlan1d(plan_bwd)");

    // optimal performance for CUFFT regarding data layout
    result = cufftSetCompatibilityMode(plan_bwd, CUFFT_COMPATIBILITY_NATIVE);
    if (result != CUFFT_SUCCESS)
      throw CUFFTError (result, "CUDA::FilterbankEngine::setup",
			"cufftSetCompatibilityMode(plan_bwd)");

    result = cufftSetStream (plan_bwd, stream);
    if (result != CUFFT_SUCCESS)
      throw CUFFTError (result, "CUDA::FilterbankEngine::setup",
			"cufftSetStream(plan_bwd)");

    DEBUG("CUDA::FilterbankEngine::setup bwd FFT plan set");
  }

  nkeep = freq_res;

  multiply.init ();
  multiply.set_nelement(nchan_subband * freq_res);

  if (filterbank->has_response())
  {
    const dsp::Response* response = filterbank->get_response();

    unsigned nchan = response->get_nchan();
    unsigned ndat = response->get_ndat();
    unsigned ndim = response->get_ndim();

    assert( nchan == filterbank->get_nchan() );
    assert( ndat == freq_res );
    assert( ndim == 2 ); // complex

    unsigned mem_size = nchan * ndat * ndim * sizeof(cufftReal);

    // allocate space for the convolution kernel
    cudaMalloc ((void**)&d_kernel, mem_size);

    nfilt_pos = response->get_impulse_pos();
    unsigned nfilt_tot = nfilt_pos + response->get_impulse_neg();

    // points kept from each small fft
    nkeep = freq_res - nfilt_tot;
 
    // copy the kernel accross
    const float* kernel = filterbank->get_response()->get_datptr(0,0);
    
    if (stream)
      cudaMemcpyAsync(d_kernel, kernel, mem_size, cudaMemcpyHostToDevice, stream);
    else
      cudaMemcpy (d_kernel, kernel, mem_size, cudaMemcpyHostToDevice);
  }

  if (!real_to_complex)
    return;
}

void CUDA::FilterbankEngine::set_scratch (float * _scratch)
{
  scratch = _scratch;
}

void CUDA::FilterbankEngine::finish ()
{
  check_error_stream ("CUDA::FilterbankEngine::finish", stream);
}


void CUDA::FilterbankEngine::perform (const dsp::TimeSeries * in, dsp::TimeSeries * out, 
            uint64_t npart, const uint64_t in_step, const uint64_t out_step)
{
  verbose = dsp::Operation::record_time || dsp::Operation::verbose;

  const unsigned npol = in->get_npol();
  const unsigned input_nchan = in->get_nchan();
  const unsigned output_nchan = out->get_nchan();
 
  // counters
  unsigned ipol, ichan;
  uint64_t ipart;
 
  // offsets into input and output
  uint64_t in_offset, out_offset;
  DEBUG("CUDA::FilterbankEngine::perform stream=" << stream);

  // GPU scratch space
  DEBUG("CUDA::FilterbankEngine::perform scratch=" << scratch);
  float2* cscratch = (float2*) scratch;

  cufftResult result;
  float * output_ptr;
  float * input_ptr;
  uint64_t output_span;

  DEBUG("CUDA::FilterbankEngine::perform input_nchan=" << input_nchan);
  DEBUG("CUDA::FilterbankEngine::perform npol=" << npol);
  DEBUG("CUDA::FilterbankEngine::perform npart=" << npart);
  DEBUG("CUDA::FilterbankEngine::perform nkeep=" << nkeep);
  DEBUG("CUDA::FilterbankEngine::perform in_step=" << in_step);
  DEBUG("CUDA::FilterbankEngine::perform out_step=" << out_step);

  for (ichan=0; ichan<input_nchan; ichan++)
  {
    for (ipol=0; ipol < npol; ipol++)
    {
      for (ipart=0; ipart < npart; ipart++)
      {
        DEBUG("CUDA::FilterbankEngine::perform ipart " << ipart << " of " << npart);

        in_offset = ipart * in_step;
        out_offset = ipart * out_step;

        DEBUG("CUDA::FilterbankEngine::perform offsets in=" << in_offset << " out=" << out_offset);

        input_ptr = const_cast<float*>(in->get_datptr (ichan, ipol)) + in_offset;

        DEBUG("CUDA::FilterbankEngine::perform FORWARD FFT inptr=" << input_ptr << " outptr=" << cscratch);
        if (real_to_complex)
        {
          result = cufftExecR2C(plan_fwd, input_ptr, cscratch);
          if (result != CUFFT_SUCCESS)
            throw CUFFTError (result, "CUDA::FilterbankEngine::perform", "cufftExecR2C");
          CHECK_ERROR ("CUDA::FilterbankEngine::perform cufftExecR2C FORWARD", stream);
        }
        else
        {
          float2* cin = (float2*) input_ptr;
          result = cufftExecC2C(plan_fwd, cin, cscratch, CUFFT_FORWARD);
          if (result != CUFFT_SUCCESS)
            throw CUFFTError (result, "CUDA::FilterbankEngine::perform", "cufftExecC2C");
          CHECK_ERROR ("CUDA::FilterbankEngine::perform cufftExecC2C FORWARD", stream);
        }

        if (d_kernel)
        {
          // complex numbers offset (d_kernel is float2*)
          unsigned offset = ichan * nchan_subband * freq_res; 
          DEBUG("CUDA::FilterbankEngine::perform multiply dedipersion kernel stream=" << stream);
          k_multiply<<<multiply.get_nblock(),multiply.get_nthread(),0,stream>>> (cscratch, d_kernel+offset);
          CHECK_ERROR ("CUDA::FilterbankEngine::perform multiply", stream);
        }

        if (plan_bwd)
        {
          DEBUG("CUDA::FilterbankEngine::perform BACKWARD FFT");
          result = cufftExecC2C (plan_bwd, cscratch, cscratch, CUFFT_INVERSE);
          if (result != CUFFT_SUCCESS)
            throw CUFFTError (result, "CUDA::FilterbankEngine::perform", "cufftExecC2C (inverse)");

          CHECK_ERROR ("CUDA::FilterbankEngine::perform cufftExecC2C BACKWARD", stream);
        }

        if (out)
        {
          output_ptr = out->get_datptr (ichan*nchan_subband, ipol) + out_offset;
          output_span = out->get_datptr (ichan*nchan_subband+1, ipol) - out->get_datptr (ichan*nchan_subband, ipol);

          const float2* input = cscratch + nfilt_pos;
          unsigned input_stride = freq_res;
          unsigned to_copy = nkeep;

          {
            dim3 threads;
            threads.x = multiply.get_nthread();

            dim3 blocks;
            blocks.x = nkeep / threads.x;
            if (nkeep % threads.x)
              blocks.x ++;

            blocks.y = nchan_subband;

            // divide by two for complex data
            float2* output_base = (float2*) output_ptr;
            unsigned output_stride = output_span / 2;

            DEBUG("CUDA::FilterbankEngine::perform output base=" << output_base << " stride=" << output_stride);
            DEBUG("CUDA::FilterbankEngine::perform input base=" << input << " stride=" << input_stride);
            DEBUG("CUDA::FilterbankEngine::perform to_copy=" << to_copy);

            k_ncopy<<<blocks,threads,0,stream>>> (output_base, output_stride,
                        input, input_stride, to_copy);
            CHECK_ERROR ("CUDA::FilterbankEngine::perform ncopy", stream);
          }
        } // if not benchmarking
      } // for each part
    } // for each polarization
  } // for each channel

  if (verbose)
    check_error_stream ("CUDA::FilterbankEngine::perform", stream);
}

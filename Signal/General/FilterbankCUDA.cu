//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// #define _DEBUG 1

#include "dsp/FilterbankCUDA.h"
#include "debug.h"

#include <cuda_runtime.h>

#include <iostream>
#include <assert.h>

void check_error (const char*);

#ifdef _DEBUG
#define CHECK_ERROR(x) check_error(x)
#else
#define CHECK_ERROR(x)
#endif


// compute 2X(w) = Z(w) + Z^*(-w) 
#define sep_X(X,z,zh) X.x = 0.5*(z.x + zh.x); X.y = 0.5*(z.y - zh.y);

// compute 2Y(w) = iZ^*(-w) - iZ(w)
#define sep_Y(Y,z,zh) Y.x = 0.5*(zh.y + z.y); Y.y = 0.5*(zh.x - z.x);

__global__ void separate (float2* d_fft, int nfft)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x + 1;
  int k = nfft - i;

  float2* p0 = d_fft;
  float2* p1 = d_fft + nfft;

  float2 p0i = p0[i];
  float2 p0k = p0[k];

  float2 p1i = p1[i];
  float2 p1k = p1[k];

  sep_X( p0[i], p0i, p1k );
  sep_X( p0[k], p0k, p1i );

  sep_Y( p1[i], p0i, p1k );
  sep_Y( p1[k], p0k, p1i );
}

__global__ void multiply (float2* d_fft, float2* kernel)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  float x = d_fft[i].x * kernel[i].x - d_fft[i].y * kernel[i].y;
  d_fft[i].y = d_fft[i].x * kernel[i].y + d_fft[i].y * kernel[i].x;
  d_fft[i].x = x;
}

__global__ void ncopy (float2* output_data, unsigned output_stride,
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

  plan_fwd = 0;
  plan_bwd = 0;
  verbose = false;
}

CUDA::FilterbankEngine::~FilterbankEngine ()
{
}

void CUDA::FilterbankEngine::setup (dsp::Filterbank* filterbank)
{
  freq_res = filterbank->get_freq_res ();
  nchan_subband = filterbank->get_nchan_subband();

  real_to_complex = (filterbank->get_input()->get_state() == Signal::Nyquist);

  DEBUG("CUDA::FilterbankEngine::setup nchan_subband=" << nchan_subband \
        << " freq_res=" << freq_res);

  DEBUG("CUDA::FilterbankEngine::setup scratch=" << scratch);

  // determine GPU capabilities 
  int device = 0;
  cudaGetDevice(&device);
  struct cudaDeviceProp device_properties;
  cudaGetDeviceProperties (&device_properties, device);
  max_threads_per_block = device_properties.maxThreadsPerBlock;

  if (real_to_complex)
  {
    DEBUG("CUDA::FilterbankEngine::setup plan size=" << freq_res*nchan_subband*2);
    cufftPlan1d (&plan_fwd, freq_res*nchan_subband*2, CUFFT_R2C, 1);
  }
  else
  {
    DEBUG("CUDA::FilterbankEngine::setup plan size=" << freq_res*nchan_subband);
    cufftPlan1d (&plan_fwd, freq_res*nchan_subband, CUFFT_C2C, 1);
  }

  DEBUG("CUDA::FilterbankEngine::setup setting stream " << stream);
  cufftSetStream (plan_fwd, stream);

  // optimal performance for CUFFT regarding data layout
  cufftSetCompatibilityMode(plan_fwd, CUFFT_COMPATIBILITY_NATIVE);

  DEBUG("CUDA::FilterbankEngine::setup fwd FFT plan set");
  if (freq_res > 1)
  {
    cufftPlan1d (&plan_bwd, freq_res, CUFFT_C2C, nchan_subband);
    cufftSetStream (plan_bwd, stream);

    // optimal performance for CUFFT regarding data layout
    cufftSetCompatibilityMode(plan_bwd, CUFFT_COMPATIBILITY_NATIVE);
    DEBUG("CUDA::FilterbankEngine::setup bwd FFT plan set");
  }

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

    cudaMemcpy (d_kernel, kernel, mem_size, cudaMemcpyHostToDevice);
  }

  if (!real_to_complex)
    return;
}

void CUDA::FilterbankEngine::set_scratch (float * _scratch)
{
  scratch = _scratch;
}

extern void check_error (const char*);

void CUDA::FilterbankEngine::finish ()
{
  check_error ("CUDA::FilterbankEngine::finish");
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

  // GPU scratch space
  DEBUG("CUDA::FilterbankEngine::perform scratch=" << scratch);
  float2* cscratch = (float2*) scratch;

  unsigned data_size = nchan_subband * freq_res;
  int threads_per_block = max_threads_per_block / 2;

  // note that each thread will set two complex numbers in each poln
  int blocks = data_size / (threads_per_block * 2);

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

        //DEBUG("CUDA::FilterbankEngine::perform offsets in=" << in_offset << " out=" << out_offset);

        input_ptr = const_cast<float*>(in->get_datptr (ichan, ipol)) + in_offset;

        //DEBUG("CUDA::FilterbankEngine::perform FORWARD FFT");
        if (real_to_complex)
        {
          cufftExecR2C(plan_fwd, input_ptr, cscratch);
          check_error ("CUDA::FilterbankEngine::perform cufftExecR2C FORWARD");
        }
        else
        {
          float2* cin = (float2*) input_ptr;
          cufftExecC2C(plan_fwd, cin, cscratch, CUFFT_FORWARD);
          check_error ("CUDA::FilterbankEngine::perform cufftExecC2C FORWARD");
        }

        blocks = data_size / threads_per_block;

        if (d_kernel)
        {
          DEBUG("CUDA::FilterbankEngine::perform multiply dedipersion kernel");
          multiply<<<blocks,threads_per_block,0,stream>>> (cscratch, d_kernel);
          check_error ("CUDA::FilterbankEngine::perform multiply");
        }

        //DEBUG("CUDA::FilterbankEngine::perform BACKWARD FFT");
        cufftExecC2C (plan_bwd, cscratch, cscratch, CUFFT_INVERSE);

        check_error ("CUDA::FilterbankEngine::perform cufftExecC2C BACKWARD");

        if (out)
        {
          output_ptr = out->get_datptr (0, ipol) + out_offset;
          output_span = out->get_datptr (1, ipol) - out->get_datptr (0, ipol);

          const float2* input = cscratch + nfilt_pos;
          unsigned input_stride = freq_res;
          unsigned to_copy = nkeep;

          {
            dim3 threads;
            threads.x = threads_per_block;

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

            ncopy<<<blocks,threads,0,stream>>> (output_base, output_stride,
                        input, input_stride, to_copy);
            check_error ("CUDA::FilterbankEngine::perform ncopy");
          }
        } // if not benchmarking
      } // for each part
    } // for each polarization
  } // for each channel

  if (verbose)
    check_error ("CUDA::FilterbankEngine::perform");
}

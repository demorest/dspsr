//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/FilterbankCUDA.h"
#include "debug.h"

#include <iostream>
using namespace std;

CUDA::Engine::Engine ()
{
  real_to_complex = false;
  nchan = 0;
  bwd_nfft = 0;

  d_fft = d_kernel = 0;

  scratch = 0;
}

CUDA::Engine::~Engine ()
{
}

void CUDA::Engine::setup (dsp::Filterbank* filterbank)
{
  bwd_nfft = filterbank->get_freq_res ();
  nchan = filterbank->get_nchan ();

  real_to_complex = (filterbank->get_input()->get_state() == Signal::Nyquist);

  twofft = false;

  DEBUG("CUDA::Engine::setup nchan=" << nchan << " bwd_nfft=" << bwd_nfft);

  unsigned data_size = nchan * bwd_nfft * 2;
  unsigned mem_size = data_size * sizeof(cufftReal);

  DEBUG("CUDA::Engine::setup data_size=" << data_size);

  // if using the twofft trick, double the forward FFT length and
  // double the number of backward FFTs

  unsigned npol = 1;
  if (twofft)
    npol = 2;

  cufftPlan1d (&plan_fwd, bwd_nfft*nchan*npol, CUFFT_C2C, 1);

  if (nchan > 1)
    cufftPlan1d (&plan_bwd, bwd_nfft, CUFFT_C2C, nchan*npol);

  if (filterbank->has_response())
  {
    // allocate space for the convolution kernel
    cudaMalloc ((void**)&d_kernel, mem_size);
 
    // copy the kernel accross
    const float* kernel = filterbank->get_response()->get_datptr(0,0);

    cudaMemcpy (d_kernel, kernel, mem_size, cudaMemcpyHostToDevice);
  }

  if (!real_to_complex || twofft)
    return;

  DEBUG("CUDA::Engine::setup real-to-complex");

  unsigned n_half = nchan * bwd_nfft / 2 + 1;
  unsigned n_half_size = n_half * sizeof(cufftReal);

  // setup realtr coeffients and variables, copy to CUDA
  double arg = 1.570796327 / (nchan * bwd_nfft); 
  double CD = 2*sin(arg)*sin(arg);
  double SD = sin(arg+arg);
  
  // used by realtr SN and CN to be copied to kernel
  std::vector<float> SN (n_half);
  std::vector<float> CN (n_half);

  SN[0]=0.0;
  CN[0]=1.0;

  for (int j=1; j<n_half; j++)
  {
    CN[j]=CN[j-1]-(CD*CN[j-1]+SD*SN[j-1]);
    SN[j]=(SD*CN[j-1]-CD*SN[j-1])+SN[j-1];
  }

  cudaMalloc((void**)&d_CN, n_half_size);
  cudaMalloc((void**)&d_SN, n_half_size);

  cudaMemcpy(d_CN,&(CN[0]),n_half_size,cudaMemcpyHostToDevice);
  cudaMemcpy(d_SN,&(SN[0]),n_half_size,cudaMemcpyHostToDevice);
}

void check_error (const char* method)
{
  cudaThreadSynchronize ();
 
  cudaError error = cudaGetLastError();
  if (error != cudaSuccess)
    throw Error (InvalidState, method, cudaGetErrorString (error));
}

#ifdef _DEBUG
#define CHECK_ERROR(x) check_error(x)
#else
#define CHECK_ERROR(x)
#endif

void CUDA::Engine::finish ()
{
  check_error ("CUDA::Engine::finish");
}

/* *************************************************************************
 *
 *
 * The twofft trick
 *
 * Where:
 *   Z = X + i Y
 *   X, Y, and Z are complex
 *   X(-w) = X*(w)
 *   Y(-w) = X*(w)
 *   Z^*(-w) = X(w) - i Y(w)
 *
 *
 ************************************************************************* */

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

/* *************************************************************************
 *
 *
 * The realtr trick
 *
 *
 ************************************************************************* */

__global__ void performRealtr (float2* d_fft, unsigned bwd_nfft,
			       float* k_SN, float* k_CN)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int k = bwd_nfft - i;
 
  float real_aa=d_fft[i].x+d_fft[k].x;
  float real_ab=d_fft[k].x-d_fft[i].x;
  
  float imag_ba=d_fft[i].y+d_fft[k].y;
  float imag_bb=d_fft[k].y-d_fft[i].y;

  float temp_real=k_CN[i]*imag_ba+k_SN[i]*real_ab;
  float temp_imag=k_SN[i]*imag_ba-k_CN[i]*real_ab;

  d_fft[k].y = -0.5*(temp_imag-imag_bb);
  d_fft[i].y = -0.5*(temp_imag+imag_bb);

  d_fft[k].x = 0.5*(real_aa-temp_real);
  d_fft[i].x = 0.5*(real_aa+temp_real);
}

/* *************************************************************************
 *
 *
 * end of tricks
 *
 *
 ************************************************************************* */

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

void CUDA::Engine::perform (const float* in)
{
  float2* cscratch = (float2*) scratch;
  float2* cin = (float2*) in;

  cufftExecC2C(plan_fwd, cin, cscratch, CUFFT_FORWARD);

  CHECK_ERROR ("CUDA::Engine::perform cufftExecC2C FORWARD");

  if (nchan == 1)
    return;

  unsigned data_size = nchan * bwd_nfft;

  int threads = 128;

  // note that each thread will set two complex numbers in each poln
  int blocks = data_size / (threads*2);

  if (real_to_complex)
  {
    DEBUG("CUDA::Engine::perform real-to-complex");

    if (twofft)
      separate<<<blocks,threads>>> (cscratch, data_size);
    else
      performRealtr<<<blocks,threads>>> (cscratch,data_size,d_SN,d_CN);

    CHECK_ERROR ("CUDA::Engine::perform separate");
  }

  blocks = data_size / threads;

  if (d_kernel)
  {
    multiply<<<blocks,threads>>> (cscratch, d_kernel);

    if (twofft)
      multiply<<<blocks,threads>>> (cscratch+data_size, d_kernel);

    CHECK_ERROR ("CUDA::Engine::perform multiply");
  }

  cufftExecC2C (plan_bwd, cscratch, cscratch, CUFFT_INVERSE);

  CHECK_ERROR ("CUDA::Engine::perform cufftExecC2C BACKWARD");

  if (!output)
    return;

  const float2* input_p0 = cscratch + nfilt_pos;
  const float2* input_p1 = input_p0 + data_size;
  unsigned input_stride = bwd_nfft;
  unsigned to_copy = nkeep;

{
  dim3 threads;
  threads.x = 128;

  dim3 blocks;
  blocks.x = nkeep / threads.x;
  if (nkeep % threads.x)
    blocks.x ++;

  blocks.y = nchan;

  // divide by two for complex data
  float2* output_base = (float2*) output;
  unsigned output_stride = output_span / 2;

  ncopy<<<blocks,threads>>> (output_base, output_stride,
			     input_p0, input_stride, to_copy);

  if (twofft)
    ncopy<<<blocks,threads>>> (output_base+output_stride/2, output_stride,
                               input_p1, input_stride, to_copy);
}

}


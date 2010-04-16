//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/FilterbankCUDA.h"

CUDA::Engine::Engine ()
{
  nchan = 0;
  bwd_nfft = 0;

  d_fft = d_kernel = 0;

  scratch = 0;
}

CUDA::Engine::~Engine ()
{
}

void CUDA::Engine::setup (unsigned _nchan, unsigned _bwd_nfft, float* _kernel) 
{
  bwd_nfft = _bwd_nfft;
  nchan = _nchan;
  kernel = _kernel;
}

void CUDA::Engine::init ()
{
  DEBUG("CUDA::Engine::init nchan=" << nchan << " bwd_nfft=" << bwd_nfft);

  unsigned data_size = nchan * bwd_nfft * 2;
  unsigned mem_size = data_size * sizeof(cufftReal);

  DEBUG("CUDA::Engine::init data_size=" << data_size);

  int device =0;
  cudaGetDevice (&device);
  cerr << "CUDA::Engine::init device: " << device << endl;

  cufftPlan1d (&plan_fwd, bwd_nfft*nchan*2, CUFFT_C2C, 1);

  cufftPlan1d (&plan_bwd, bwd_nfft, CUFFT_C2C, nchan*2);

  // allocate space for the convolution kernel
  cudaMalloc ((void**)&d_kernel, mem_size);
 
  // copy the kernel accross
  cudaMemcpy (d_kernel, kernel, mem_size, cudaMemcpyHostToDevice);
}

// compute Z(w) + Z^*(-w) 
#define sep_p0(p0,z,zh) p0.x = z.x + zh.x; p0.y = z.y - zh.y;

// compute iZ^*(-w) - iZ(w)
#define sep_p1(p1,z,zh) p1.x = zh.y + z.y; p1.y = zh.x - z.x;

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

  sep_p0( p0[i], p0i, p1k );
  sep_p0( p0[k], p0i, p1i );

  sep_p1( p1[i], p0i, p1k );
  sep_p1( p1[k], p0i, p1i );
}

__global__ void multiply (float2* d_fft, float2* kernel)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  float x = d_fft[i].x * kernel[i].x - d_fft[i].y * kernel[i].y;
  d_fft[i].y = d_fft[i].x * kernel[i].y + d_fft[i].y * kernel[i].x;
  d_fft[i].x = x;
}

typedef struct { float2 p0; float2 p1 } evect;

__global__ void merge (evect* into_data, unsigned into_stride, 
		       const float2* p0, const float2* p1, unsigned stride,
		       unsigned to_copy)
{
  unsigned datum = blockIdx.x*blockDim.x+threadIdx.x;

  into_data += blockIdx.x * into_stride;
  from_p0 += blockIdx.x * from_stride;
  from_p1 += blockIdx.x * from_stride;

  evect result = { p0[threadIdx.x], p1[threadIdx.x] };
  into_data[threadIdx.x] = result;
}



void CUDA::Engine::perform (const float* in)
{
  if (!d_kernel)
    init ();

  float2* cscratch = (float2*) scratch;
  float2* cin = (float2*) in;

  cufftExecC2C(plan_fwd, cin, cscratch, CUFFT_FORWARD);

  unsigned data_size = nchan * bwd_nfft;

  int blocks = 256;
  int threads = data_size / (blocks*2);

  separate<<<threads,blocks>>> (cscratch, data_size);
 
  int threads = data_size / blocks;

  performConvCUDA<<<threads,blocks>>> (cscratch, d_kernel);
  performConvCUDA<<<threads,blocks>>> (cscratch+data_size, d_kernel);
  
  cufftExecC2C (plan_bwd, cscratch, cscratch, CUFFT_INVERSE);

  unsigned nchan = output_ptr.size();
  blocks = nchan;
  threads = nkeep;

  const float2* from_p0 = cscratch + nfilt_pos;
  const float2* from_p1 = from_p0 + data_size;
  unsigned from_stride = bwd_nfft;

  evect* into_base = (evect*) output_ptr[0];
  evect* into_next = (evect*) output_ptr[1];

  unsigned into_stride = into_next - into_data;
  unsigned to_copy = nkeep;

  merge<<<threads,blocks>>> (into_base, into_stride,
			     from_p0, from_p1, from_stride, to_copy);
}





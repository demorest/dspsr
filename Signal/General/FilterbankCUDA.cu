//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten and Jonathon Kocz
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// #define _RECORD_EVENTS

#include "dsp/FilterbankCUDA.h"

//#define _DEBUG
#include "debug.h"

#ifdef _RECORD_EVENTS
#include <iostream>
#endif

using namespace std;

CUDA::elapsed::elapsed ()
{
  cudaEventCreate (&after);
  total = 0;
}

void CUDA::elapsed::wrt (cudaEvent_t before)
{
  float time;
  cudaEventSynchronize (after);
  cudaEventElapsedTime (&time, before, after);
  total += time;
}

CUDA::Engine::Engine ()
{
  timers = 0;

  nchan = 0;
  bwd_nfft = 0;

  d_fft = d_kernel = 0;

  scratch = 0;
}

CUDA::Engine::~Engine ()
{
#ifdef _RECORD_EVENTS
  cerr << "******************************************************" << endl;
  cerr << "memcpy to device: " << timers->copy_to.total << " ms" << endl;
  cerr << "forward FFT: " << timers->fwd.total << " ms" << endl;
  cerr << "realtr: " << timers->realtr.total << " ms" << endl;
  cerr << "conv: " << timers->conv.total << " ms" << endl;
  cerr << "backward FFT: " << timers->bwd.total << " ms" << endl;
  cerr << "memcpy from device: " << timers->copy_from.total << " ms" << endl;
  cerr << endl;
#endif
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

  cufftPlan1d (&plan_fwd, bwd_nfft*nchan, CUFFT_C2C, 1);

  cufftPlan1d (&plan_bwd, bwd_nfft, CUFFT_C2C, nchan);

  // allocate space for the convolution kernel
  cudaMalloc((void**)&d_kernel, mem_size);
 
  // copy the kernel accross
  cudaMemcpy(d_kernel,kernel,mem_size,cudaMemcpyHostToDevice);

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

#ifdef _RECORD_EVENTS
  cudaEventCreate (&start);
  timers = new Timers;
#endif

}



__global__ void performRealtr (float2* d_fft, unsigned bwd_nfft,
			       float* k_SN, float* k_CN)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int k = bwd_nfft - i;
 
  float real_aa, real_ab, imag_ba, imag_bb, temp_real, temp_imag;

  // eg Final result = 16 elements. N = 8, NK = 8, NH = 4;

  //fprintf(stderr,"i : %d\n", i);
  //fprintf(stderr,"k : %d\n", k);
 
  
  real_aa=d_fft[i].x+d_fft[k].x;
  real_ab=d_fft[k].x-d_fft[i].x;
  
  imag_ba=d_fft[i].y+d_fft[k].y;
  imag_bb=d_fft[k].y-d_fft[i].y;

  temp_real=k_CN[i]*imag_ba+k_SN[i]*real_ab;
  temp_imag=k_SN[i]*imag_ba-k_CN[i]*real_ab;

  d_fft[k].y = -0.5*(temp_imag-imag_bb);
  d_fft[i].y = -0.5*(temp_imag+imag_bb);

  d_fft[k].x = 0.5*(real_aa-temp_real);
  d_fft[i].x = 0.5*(real_aa+temp_real);
}

__global__ void performConvCUDA (float2* d_fft, float2* kernel)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  float x = d_fft[i].x * kernel[i].x - d_fft[i].y * kernel[i].y;
  d_fft[i].y = d_fft[i].x * kernel[i].y + d_fft[i].y * kernel[i].x;
  d_fft[i].x = x;
}

__global__ void performPrependCUDA (const float2* from_data,float2* into_data,unsigned into_stride,  unsigned from_stride, unsigned to_copy)
{
  unsigned datum = blockIdx.x*blockDim.x+threadIdx.x;

 unsigned block = datum / to_copy;
 unsigned index = datum - block* to_copy;
 into_data += block * into_stride;
 from_data += block * from_stride;

 into_data[index] = from_data[index];  
}



void CUDA::Engine::perform (const float* in)
{
  if (!d_kernel)
    init ();

  int device =9;
  cudaGetDevice (&device);
  //cerr << "CUDA::Engine::perform device: " << device << endl;

  cudaThreadSynchronize ();
 
 cudaError error = cudaGetLastError();
  if (error != cudaSuccess)
    cerr << "BEFORE CUFFT_FORWARD: " << cudaGetErrorString (error) << " device: " << device << endl;

  //cerr << "CUDA::Engine::perform scratch=" << scratch << endl;

  float2* cscratch = (float2*) scratch;
  float2* cin = (float2*) in;

  cufftExecC2C(plan_fwd, cin, cscratch, CUFFT_FORWARD);
  //cufftExecC2C(plan_fwd, cscratch, cscratch, CUFFT_FORWARD);

  cudaThreadSynchronize ();
   error = cudaGetLastError();
  if (error != cudaSuccess)
    cerr << "FAIL CUFFT_FORWARD: " << cudaGetErrorString (error) << " device: " << device << endl;

  int Blks = 256;
  unsigned data_size = nchan * bwd_nfft;
  int realtrThread = data_size / (Blks*2);

  performRealtr<<<realtrThread,Blks>>>(cscratch,data_size,d_SN,d_CN);
 
  cudaThreadSynchronize ();

  error = cudaGetLastError();
  if (error != cudaSuccess)
    cerr << "FAIL REALTR: " << cudaGetErrorString (error) << endl;
  
  
  int BlkThread = data_size / Blks;

  performConvCUDA<<<BlkThread,Blks>>>(cscratch,d_kernel);
  
  cudaThreadSynchronize();

  error = cudaGetLastError();
  if (error != cudaSuccess)
    cerr << "FAIL ConvCUDA: " << cudaGetErrorString (error) << endl;

  cufftExecC2C (plan_bwd, cscratch, cscratch, CUFFT_INVERSE);

  cudaThreadSynchronize ();

  error = cudaGetLastError();
  if (error != cudaSuccess)
    cerr << "FAIL CUFFT_INVERSE: " << cudaGetErrorString (error) << endl;
  

  unsigned nchan = output_ptr.size();
  int prependThread = nchan*nkeep / Blks;


  const float2* from_data = cscratch + nfilt_pos;
  unsigned from_stride = freq_res;
  float2* into_data = (float2*) output_ptr[0];
  unsigned into_stride = (output_ptr[1] - output_ptr[0]) / 2;
  unsigned to_copy = nkeep;

  performPrependCUDA<<<prependThread,Blks>>>(from_data, into_data,into_stride,from_stride,to_copy);


  // Replaced by performPrependCUDA
  /* for (unsigned ichan=0; ichan < nchan; ichan++)
  {
    //cerr << "nchan :" << nchan << endl;
    float* c_time = scratch + ichan*freq_res*2;
    float* data_into = output_ptr[ichan];
    const float* data_from = c_time + nfilt_pos*2;  // complex nos.

    cudaError err = cudaMemcpy (data_into, data_from, nkeep * 2 * sizeof(float), cudaMemcpyDeviceToDevice);
    if (error != cudaSuccess)
    cerr << "FAIL MEMCPY: " << cudaGetErrorString (error) << " device: " << device << endl;
  
    }*/
}





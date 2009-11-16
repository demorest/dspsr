//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten and Jonathon Kocz
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#define _RECORD_EVENTS

#include "dsp/FilterbankCUDA.h"

//#define _DEBUG
#include "debug.h"

#ifdef _RECORD_EVENTS
#include <iostream>
using namespace std;
#endif

CUDA::elapsed::elapsed ()
{
  cudaEventCreate (&after);
  total = 0;
}

void CUDA::elapsed::wrt (cudaEvent_t before)
{
  float time;
  cudaEventSynchronize (after);
  cutilSafeCall( cudaEventElapsedTime (&time, before, after) );
  total += time;
}

CUDA::Engine::Engine (int _device)
{
  device = _device;

  timers = 0;

  nchan = 0;
  bwd_nfft = 0;

  d_fft = d_kernel = 0;
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
  int ndevice = 0;
  cutilSafeCall( cudaGetDeviceCount(&ndevice) );

  if (device >= ndevice)
    throw Error (InvalidParam, "CUDA::Engine::init",
		 "device=%d >= ndevice=%d", device, ndevice);

  cutilSafeCall( cudaSetDevice(device) );

  DEBUG("CUDA::Engine::init nchan=" << nchan << " bwd_nfft=" << bwd_nfft);

  unsigned data_size = nchan * bwd_nfft * 2;
  unsigned mem_size = data_size * sizeof(cufftReal);

  DEBUG("CUDA::Engine::init data_size=" << data_size);

  cutilSafeCall( cudaStreamCreate(&stream) );

  cutilSafeCall( cudaMallocHost((void**)&pinned, mem_size) );

  cutilSafeCall( cudaMalloc((void**)&d_fft, mem_size + 2*sizeof(cufftReal)) );

  // one forward big FFT
  cufftSafeCall( cufftPlan1d (&plan_fwd, bwd_nfft*nchan, CUFFT_C2C, 1) );
  cufftSafeCall( cufftSetStream (plan_fwd, stream) );

  // nchan backward little FFTs
  cufftSafeCall( cufftPlan1d (&plan_bwd, bwd_nfft, CUFFT_C2C, nchan) );
  cufftSafeCall( cufftSetStream (plan_bwd, stream) );

  // allocate space for the convolution kernel
  cutilSafeCall( cudaMalloc((void**)&d_kernel, mem_size) );
 
  // copy the kernel accross
  cutilSafeCall( cudaMemcpy(d_kernel,kernel,mem_size,cudaMemcpyHostToDevice) );

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

  cutilSafeCall( cudaMalloc((void**)&d_CN, n_half_size) );
  cutilSafeCall( cudaMalloc((void**)&d_SN, n_half_size) );

  cutilSafeCall( cudaMemcpy(d_CN,&(CN[0]),n_half_size,cudaMemcpyHostToDevice));
  cutilSafeCall( cudaMemcpy(d_SN,&(SN[0]),n_half_size,cudaMemcpyHostToDevice));

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

void CUDA::Engine::perform (const float* in, float* out)
{
  if (!d_kernel)
    init ();

  unsigned mem_size = bwd_nfft * nchan * 2 * sizeof(float);

  memcpy (pinned, in, mem_size);

  DEBUG("CUDA::Engine::perform d_fft=" << d_fft << " in=" << in);

#ifdef _RECORD_EVENTS
  cudaEventRecord (start, stream);
#endif

  cutilSafeCall(cudaMemcpyAsync( d_fft, pinned, mem_size,
				 cudaMemcpyHostToDevice, stream ));

#ifdef _RECORD_EVENTS
  cudaEventRecord (timers->copy_to.after, stream);
#endif

  cufftSafeCall (cufftExecC2C(plan_fwd, d_fft, d_fft, CUFFT_FORWARD));

#ifdef _RECORD_EVENTS
  cudaEventRecord (timers->fwd.after, stream);
#endif

  int Blks = 256;
  unsigned data_size = nchan * bwd_nfft;
  int realtrThread = data_size / (Blks*2);

  performRealtr<<<realtrThread,Blks,0,stream>>>(d_fft,data_size,d_SN,d_CN);

#ifdef _RECORD_EVENTS
  cudaEventRecord (timers->realtr.after, stream);
#endif

  int BlkThread = data_size / Blks;

  performConvCUDA<<<BlkThread,Blks,0,stream>>>(d_fft,d_kernel);

#ifdef _RECORD_EVENTS
  cudaEventRecord (timers->conv.after, stream);
#endif

  cufftSafeCall(cufftExecC2C(plan_bwd, d_fft, d_fft, CUFFT_INVERSE));

#ifdef _RECORD_EVENTS
  cudaEventRecord (timers->bwd.after, stream);
#endif

  cutilSafeCall(cudaMemcpyAsync( pinned, d_fft, mem_size,
				 cudaMemcpyDeviceToHost, stream ));

#ifdef _RECORD_EVENTS
  cudaEventRecord (timers->copy_from.after, stream);
#endif

  DEBUG("CUDA::Engine::wait call cudaStreamSynchronize");

  cutilSafeCall(cudaStreamSynchronize (stream));

#ifdef _RECORD_EVENTS
  cudaEventSynchronize (start);
  timers->copy_to.wrt (start);
  timers->fwd.wrt (timers->copy_to.after);
  timers->realtr.wrt (timers->fwd.after);
  timers->conv.wrt (timers->realtr.after);
  timers->bwd.wrt (timers->conv.after);
  timers->copy_from.wrt (timers->bwd.after);
#endif

  DEBUG("CUDA::Engine::wait memcpy result from pinned");

  memcpy (out, pinned, mem_size);
}

//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten and Jonathon Kocz
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/FilterbankCUDA.h"

__global__ void performConvCUDA (float2*, float2*);	
__global__ void performRealtr (float2*, unsigned, float*, float*);

void CUDA::Filterbank::setup (unsigned nchan, unsigned bwd_nfft, float* kernel) 
{
  if (stream.size() == nstream)
    return;

  Stream* str = new Stream (nchan, bwd_nfft, kernel);
  add_stream (str);

  for (unsigned i=0; i<nstream; i++)
    add_stream ( new Stream (*str) );
}

CUDA::Filterbank::Stream* CUDA::Filterbank::get_stream (unsigned i)
{
  return dynamic_cast<Stream*>( stream[i].get() );
}

void CUDA::Filterbank::run ()
{
  if (stream.size() == 0)
    throw Error (InvalidState, "CUDA::Filterbank::run", "no streams");

  for (unsigned i=0; i<stream.size(); i++)
    get_stream(i)->forward_fft ();

  for (unsigned i=0; i<stream.size(); i++)
    get_stream(i)->realtr ();

  for (unsigned i=0; i<stream.size(); i++)
    get_stream(i)->convolve ();

  for (unsigned i=0; i<stream.size(); i++)
    get_stream(i)->backward_fft ();

  for (unsigned i=0; i<stream.size(); i++)
    get_stream(i)->retrieve ();
}

void CUDA::Filterbank::Stream::init ()
{
  unsigned data_size = nchan * bwd_nfft * 2;
  unsigned mem_size = data_size * sizeof(cufftReal);

  cutilSafeCall( cudaStreamCreate(&stream) );

  cutilSafeCall(cudaMalloc((void**)&d_in, mem_size)); 

  cutilSafeCall(cudaMalloc((void**)&d_out, mem_size + 2*sizeof(cufftReal)));
}

void CUDA::Filterbank::Stream::zero ()
{
  d_in = d_out = d_kernel = 0;
}

CUDA::Filterbank::Stream::Stream (const CUDA::Filterbank::Stream* copy)
{
  zero ();

  copy = _copy;
  kernel = 0;
}

void CUDA::Filterbank::Stream::copy_init ()
{
  bwd_nfft = copy->bwd_nfft;
  nchan = copy->nchan;

  init ();

  plan_fwd = copy->plan_fwd;
  plan_bwd = copy->plan_bwd;
  d_kernel = copy->d_kernel;
  d_SN = copy->d_SN;
  d_CN = copy->d_CN;
}

CUDA::Filterbank::Stream::Stream (unsigned _nchan,
                                unsigned _bwd_nfft, 
			        float* _kernel)
{
  zero ();

  bwd_nfft = _bwd_nfft;
  nchan = _nchan;
  kernel = _kernel;
}

void CUDA::Filterbank::Stream::work_init ()
{
  init ();

  unsigned data_size = nchan * bwd_nfft * 2;
  unsigned mem_size = data_size * sizeof(cufftReal);

  // allocate space for the convolution kernel
  cutilSafeCall(cudaMalloc((void**)&d_kernel, mem_size));
 
  // copy the kernel accross
  cutilSafeCall(cudaMemcpy(d_kernel,kernel,mem_size,cudaMemcpyHostToDevice));

  // one forward big FFT
  cufftSafeCall(cufftPlan1d(&plan_fwd, bwd_nfft*nchan, CUFFT_C2C, 1));
  // nchan backward little FFTs
  cufftSafeCall(cufftPlan1d(&plan_bwd, bwd_nfft, CUFFT_C2C, nchan));

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

  cutilSafeCall(cudaMalloc((void**)&d_CN, n_half_size));
  cutilSafeCall(cudaMalloc((void**)&d_SN, n_half_size));

  cutilSafeCall(cudaMemcpy(d_CN,&(CN[0]),n_half_size,cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(d_SN,&(SN[0]),n_half_size,cudaMemcpyHostToDevice));
}

void CUDA::Filterbank::Stream::queue ()
{
  if (!d_kernel)
  {
    if (copy)
      copy_init ();
    else
      work_init ();
  }

  Job* my_job = static_cast<Job*> (job);

  unsigned mem_size = bwd_nfft * nchan * 2 * sizeof(float);

  cutilSafeCall(cudaMemcpyAsync( d_in, my_job->in, mem_size,
				 cudaMemcpyHostToDevice, stream ));
}

void CUDA::Filterbank::Stream::wait ()
{

}

void CUDA::Filterbank::Stream::forward_fft ()
{
  cufftSetStream (plan_fwd, stream);
  cufftSafeCall(cufftExecC2C(plan_fwd, d_in, d_out,CUFFT_FORWARD));
}

void CUDA::Filterbank::Stream::realtr ()
{
  int Blks = 256;
  unsigned data_size = nchan * bwd_nfft;
  int realtrThread = data_size / (Blks*2);

  performRealtr<<<realtrThread+1,Blks,0,stream>>>(d_out,data_size,d_SN,d_CN);
}

void CUDA::Filterbank::Stream::convolve ()
{
  int Blks = 256;
  unsigned data_size = nchan * bwd_nfft;
  int BlkThread = data_size / Blks;

  performConvCUDA<<<BlkThread,Blks,0,stream>>>(d_out,d_kernel);
}

void CUDA::Filterbank::Stream::backward_fft ()
{
  cufftSetStream (plan_bwd, stream);
  cufftSafeCall(cufftExecC2C(plan_bwd, d_out, d_out, CUFFT_INVERSE));
}

void CUDA::Filterbank::Stream::retrieve ()
{
  Job* my_job = static_cast<Job*>(job);

  unsigned data_size = nchan * bwd_nfft * 2;
  unsigned mem_size = data_size * sizeof(cufftReal);

  cutilSafeCall(cudaMemcpyAsync( my_job->out, d_out, mem_size,
				 cudaMemcpyDeviceToHost, stream ));
}

void CUDA::Filterbank::Stream::run ()
{
  throw Error (InvalidState, "CUDA::Filterbank::Stream::run",
	       "should not be called directly");
}


__global__ void performRealtr (float2* d_out, unsigned bwd_nfft,
			       float* k_SN, float* k_CN)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int k = bwd_nfft - i;
 
  float real_aa, real_ab, imag_ba, imag_bb, temp_real, temp_imag;

  // eg Final result = 16 elements. N = 8, NK = 8, NH = 4;

  //fprintf(stderr,"i : %d\n", i);
  //fprintf(stderr,"k : %d\n", k);
 
  
  real_aa=d_out[i].x+d_out[k].x;
  real_ab=d_out[k].x-d_out[i].x;
  
  imag_ba=d_out[i].y+d_out[k].y;
  imag_bb=d_out[k].y-d_out[i].y;

  temp_real=k_CN[i]*imag_ba+k_SN[i]*real_ab;
  temp_imag=k_SN[i]*imag_ba-k_CN[i]*real_ab;

  d_out[k].y = (-1)*(temp_imag-imag_bb)/2;
  d_out[i].y = (-1)*(temp_imag+imag_bb)/2;

  d_out[k].x = (real_aa-temp_real)/2;
  d_out[i].x = (real_aa+temp_real)/2;
  
  // d_out[i].x = 0;
  //d_out[k].x = 0;
  //d_out[i].y = 0;
  //d_out[k].y = 0;

}

__global__ void performConvCUDA (float2* d_out,float2* kernel_data)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  float x = d_out[i].x * kernel_data[i].x - d_out[i].y * kernel_data[i].y;
  d_out[i].y = d_out[i].x * kernel_data[i].y + d_out[i].y * kernel_data[i].x;
  d_out[i].x = x;
}

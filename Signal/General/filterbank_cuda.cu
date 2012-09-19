//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

 #define _DEBUG 1

#include "dsp/filterbank_engine.h"
#include "dsp/filterbank_cuda.h"
#include "debug.h"
#include <iostream>
using namespace std;

void check_error (const char*);

#ifdef _DEBUG
#define CHECK_ERROR(x) check_error(x)
#else
#define CHECK_ERROR(x)
#endif
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
  output_data += blockIdx.y * output_stride; //blockIdx.y = nchan these will both be zero for inchan=outchan
  input_data += blockIdx.y * input_stride;

  unsigned index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < to_copy)
    output_data[index] = input_data[index];
}

void filterbank_cuda_perform (filterbank_engine* engine, 
			      filterbank_cuda* cuda,
			      const float* in, 
            const int max_threads_per_block)
{
  float2* cscratch = (float2*) engine->scratch;

  unsigned data_size = engine->nchan * engine->freq_res; //cuda->bwd_nfft;
  int threads_per_block = max_threads_per_block / 2;

  // note that each thread will set two complex numbers in each poln
  // This must be refering to the real to complex stuff... ignore, this is not used for c2c
  int blocks = data_size / (threads_per_block * 2);

  if (in)
  {
    if (cuda->real_to_complex)
    {
      float * cin = (float *) in;
      cufftExecR2C(cuda->plan_fwd, cin, cscratch);
      CHECK_ERROR ("CUDA::FilterbankEngine::perform cufftExecR2C FORWARD");
    }
    else
    {
      float2* cin = (float2*) in;
      cufftExecC2C(cuda->plan_fwd, cin, cscratch, CUFFT_FORWARD);
      CHECK_ERROR ("CUDA::FilterbankEngine::perform cufftExecR2C FORWARD");
    }
  } else {
	  cerr << "CUDA::FilterbankEngine::perform: NO INPUT!" << endl;
  }

  blocks = data_size / threads_per_block;

//  cerr << "CUDA::FilterbankEngine::perform datasize=" << data_size << " blocks=" << blocks << endl;

  if (cuda->d_kernel)
  {
    multiply<<<blocks,threads_per_block,0,cuda->stream>>> (cscratch, cuda->d_kernel);
    CHECK_ERROR ("CUDA::FilterbankEngine::perform multiply");
  }

  cufftExecC2C (cuda->plan_bwd, cscratch, cscratch, CUFFT_INVERSE);

  CHECK_ERROR ("CUDA::FilterbankEngine::perform cufftExecC2C BACKWARD");

  if (!engine->output){
	  cerr << "CUDA::FilterbankEngine::perform NO OUTPUT!" << endl;
    return;
  }

  const float2* input = cscratch + engine->nfilt_pos;
  unsigned input_stride = engine->freq_res;
  //unsigned input_stride = cuda->bwd_nfft;
  unsigned to_copy = engine->nkeep;

  {
    dim3 threads;
    threads.x = threads_per_block;

    dim3 blocks;
    blocks.x = engine->nkeep / threads.x;
    if (engine->nkeep % threads.x)
      blocks.x ++;

    blocks.y = engine->nchan; // this will be 1 for input chan == output chan
    
    // divide by two for complex data
    float2* output_base = (float2*) engine->output;
    unsigned output_stride = engine->output_span / 2;
    if (cuda->verbose) {
    	cerr << "copy: blocks.x=" << blocks.x << " blocks.y=" << blocks.y;
    	cerr << " output_base=" << output_base << " output stride=" << output_stride << " input=" << input << " input stride=" << input_stride << " tocopy=" << to_copy << endl;
    }
    ncopy<<<blocks,threads,0,cuda->stream>>> (output_base, output_stride,
					      input, input_stride, to_copy);
  }
  
  if (cuda->verbose)
    check_error ("CUDA::FilterbankEngine::perform at the end");
}

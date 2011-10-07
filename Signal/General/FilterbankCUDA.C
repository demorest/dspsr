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
using namespace std;

CUDA::FilterbankEngine::FilterbankEngine (cudaStream_t _stream)
{
  real_to_complex = false;
  nchan = 0;
  bwd_nfft = 0;

  d_fft = d_kernel = 0;

  scratch = 0;

  stream = _stream;

  plan_fwd = 0;
  plan_bwd = 0;
}

CUDA::FilterbankEngine::~FilterbankEngine ()
{
}

void CUDA::FilterbankEngine::setup (dsp::Filterbank* filterbank)
{
  bwd_nfft = filterbank->get_freq_res ();
  nchan = filterbank->get_nchan ();

  real_to_complex = (filterbank->get_input()->get_state() == Signal::Nyquist);

  twofft = false;

  DEBUG("CUDA::FilterbankEngine::setup nchan=" << nchan \
	<< " bwd_nfft=" << bwd_nfft);

  unsigned data_size = nchan * bwd_nfft * 2;
  unsigned mem_size = data_size * sizeof(cufftReal);

  DEBUG("CUDA::FilterbankEngine::setup data_size=" << data_size);

  // if using the twofft trick, double the forward FFT length and
  // double the number of backward FFTs

  unsigned npol = 1;
  if (twofft)
    npol = 2;

  DEBUG("CUDA::FilterbankEngine::setup plan size=" << bwd_nfft*nchan*npol);
  cufftPlan1d (&plan_fwd, bwd_nfft*nchan*npol, CUFFT_C2C, 1);
  DEBUG("CUDA::FilterbankEngine::setup setting stream" << stream);
  cufftSetStream (plan_fwd, stream);

  DEBUG("CUDA::FilterbankEngine::setup fwd FFT plan set");
  if (nchan > 1)
  {
    cufftPlan1d (&plan_bwd, bwd_nfft, CUFFT_C2C, nchan*npol);
    cufftSetStream (plan_bwd, stream);
  }

  DEBUG("CUDA::FilterbankEngine::setup bwd FFT plan set");

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

  DEBUG("CUDA::FilterbankEngine::setup real-to-complex");

  unsigned nfft = nchan * bwd_nfft;
  unsigned n_half = nfft / 2 + 1;
  unsigned n_half_size = n_half * sizeof(cufftReal);

  // used by realtr SN and CN to be copied to kernel
  std::vector<float> SN (n_half);
  std::vector<float> CN (n_half);

  SN[0]=0.0;
  CN[0]=1.0;

  for (unsigned j=1; j<n_half; j++)
  {
    CN[j] = cos (j*M_PI/nfft);
    SN[j] = sin (j*M_PI/nfft);
  }

  cudaMalloc((void**)&d_CN, n_half_size);
  cudaMalloc((void**)&d_SN, n_half_size);

  cudaMemcpy(d_CN,&(CN[0]),n_half_size,cudaMemcpyHostToDevice);
  cudaMemcpy(d_SN,&(SN[0]),n_half_size,cudaMemcpyHostToDevice);
}



void CUDA::FilterbankEngine::finish ()
{
  //check_error ("CUDA::FilterbankEngine::finish");
}


void CUDA::FilterbankEngine::perform (const float* in)
{
  verbose = dsp::Operation::record_time || dsp::Operation::verbose;

  /*
    CUDA::FilterbankEngine multiply inherits from

    filterbank_engine - the first argument
    filterbank_cuda - the second argument
    Reference::Able - not an argument

    The implicit casts performed on the following line will work.
    The relative offsets with respect to this are applied.
  */

  filterbank_cuda_perform (this, this, in);
}

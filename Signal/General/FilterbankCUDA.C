//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#define _DEBUG 1

#include "dsp/FilterbankCUDA.h"
#include "debug.h"

#include <cuda_runtime.h>

#include <iostream>
using namespace std;

CUDA::FilterbankEngine::FilterbankEngine (cudaStream_t _stream)
{
  real_to_complex = false;
  nchan = 0;
//  bwd_nfft = 0;
  freq_res =0;

  d_fft = d_kernel = 0;

  scratch = 0;

  stream = _stream;

  plan_fwd = 0;
  plan_bwd = 0;
  verbose = false;
}

CUDA::FilterbankEngine::~FilterbankEngine ()
{
	cerr << "CUDA::FilterbankEngine::~FilterbankEngine" <<endl;
	if (d_CN) {
		cudaFree(d_CN);
		cerr << "CUDA::FilterbankEngine::~FilterbankEngine freed CN" << endl;
	}
	if (d_SN) {
		cudaFree(d_SN);
		cerr << "CUDA::FilterbankEngine::~FilterbankEngine freed SN" << endl;
	}
	if (d_kernel) {
		cudaFree(d_kernel);
		cerr << "CUDA::FilterbankEngine::~FilterbankEngine freed kernel" << endl;
	}
}

void CUDA::FilterbankEngine::setup (dsp::Filterbank* filterbank)
{
	/*
	 * engine->perform is called for each input chnanel and each input polarization,
	 * but this setup function is only called once
	 *
	 * for each input channel, we need to perform a nchan_subband*freq_res forward fft,
	 * then multiply by kernel and do nchan_subband ffts each of size freq_res
	 *
	 * class attributes here are declared in the struct in filterbank_cuda.h which are then 'inherited'
	 * in FilterbankCUDA.h
	 */
  freq_res = filterbank->get_freq_res ();
  nchan_subband = filterbank->get_nchan_subband();
  nchan = filterbank->get_nchan () / filterbank->get_input()->get_nchan(); // GJ added input.nchan

  real_to_complex = (filterbank->get_input()->get_state() == Signal::Nyquist);


  DEBUG("CUDA::FilterbankEngine::setup nchan=" << nchan << " nchan_subband=" << nchan_subband\
	<< " freq_res=" << freq_res);

  unsigned data_size = nchan_subband * freq_res * 2;
  unsigned mem_size = data_size * sizeof(cufftReal);

  // determine GPU capabilities 
  int device = 0;
  cudaGetDevice(&device);
  struct cudaDeviceProp device_properties;
  cudaGetDeviceProperties (&device_properties, device);
  max_threads_per_block = device_properties.maxThreadsPerBlock;

  DEBUG("CUDA::FilterbankEngine::setup data_size=" << data_size);

  if (real_to_complex)
  {
    DEBUG("CUDA::FilterbankEngine::setup plan size=" << freq_res*nchan*2);
    cufftPlan1d (&plan_fwd, freq_res*nchan*2, CUFFT_R2C, 1);
  }
  else
  {
    DEBUG("CUDA::FilterbankEngine::setup plan size=" << freq_res*nchan);
    cufftPlan1d (&plan_fwd, freq_res*nchan, CUFFT_C2C, 1);
  }

  DEBUG("CUDA::FilterbankEngine::setup setting stream" << stream);
  cufftSetStream (plan_fwd, stream);

  // optimal performance for CUFFT regarding data layout
  cufftSetCompatibilityMode(plan_fwd, CUFFT_COMPATIBILITY_NATIVE);

  DEBUG("CUDA::FilterbankEngine::setup fwd FFT plan set");

  if (freq_res > 1)
  {
    if(cufftPlan1d (&plan_bwd, freq_res, CUFFT_C2C, nchan_subband)) {
    		      throw Error (InvalidState, "dsp::FilterbankEngine:setup bad bwd_fft plan"
    		                   );
    		  }
    cufftSetStream (plan_bwd, stream);
  }

  // optimal performance for CUFFT regarding data layout
  cufftSetCompatibilityMode(plan_bwd, CUFFT_COMPATIBILITY_NATIVE);
  DEBUG("CUDA::FilterbankEngine::setup bwd FFT plan set=" << plan_bwd);

  if (filterbank->has_response())
  {
    // allocate space for the convolution kernel
	  // This size (mem_size) is correct because each kernel contains nchan_subband kernels each of length freq_res
    cudaMalloc ((void**)&d_kernel, mem_size);
 
    // copy the kernel accross
    //This now happens later
//    const float* kernel = filterbank->get_response()->get_datptr(0,0); // how do we deal with this, we need to get the right kernel

//    cudaMemcpy (d_kernel, kernel, mem_size, cudaMemcpyHostToDevice);
  }

  if (!real_to_complex)
    return;

}


void CUDA::FilterbankEngine::sendKernel(dsp::Filterbank* filterbank, unsigned _ichan)
{
	unsigned data_size = nchan_subband * freq_res * 2;
	unsigned mem_size = data_size * sizeof(cufftReal);

	if (filterbank->has_response())
	{
		if (verbose) {
			cerr << "have response, sending kernel for input channel=" <<_ichan <<
					"response channel starting at=" << _ichan*nchan_subband << endl;
			cerr << "response has nchan=" << filterbank->get_response()->get_nchan() << endl;
		}
	// copy the kernel accross
	const float* kernel = filterbank->get_response()->get_datptr(_ichan*nchan_subband,0); //send kernels for input channel _ichan

	cudaMemcpy (d_kernel, kernel, mem_size, cudaMemcpyHostToDevice);
	}

}
void CUDA::FilterbankEngine::finish ()
{
//  check_error ("CUDA::FilterbankEngine::finish");
	if (verbose) {
		cerr << "CUDA::FilterbankEngine::finish" << endl;
	}
}


void CUDA::FilterbankEngine::perform (const float* in)
{
//  verbose = dsp::Operation::record_time || dsp::Operation::verbose;

  /*
    CUDA::FilterbankEngine multiply inherits from

    filterbank_engine - the first argument
    filterbank_cuda - the second argument
    Reference::Able - not an argument

    The implicit casts performed on the following line will work.
    The relative offsets with respect to this are applied.
  */
  if (verbose) {
	  cerr << "perform: engine nchan=" << nchan
			  << " nchan_subband=" << nchan_subband << endl;
  }
  filterbank_cuda_perform (this, this, in, max_threads_per_block);
}

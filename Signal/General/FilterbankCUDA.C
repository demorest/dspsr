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

  DEBUG("CUDA::FilterbankEngine::setup data_size=" << data_size);

  // if using the twofft trick, double the forward FFT length and
  // double the number of backward FFTs

  unsigned npol = 1;

  DEBUG("CUDA::FilterbankEngine::setup plan size=" << freq_res*nchan_subband);
  cufftPlan1d (&plan_fwd, freq_res*nchan_subband, CUFFT_C2C, 1);
  DEBUG("CUDA::FilterbankEngine::setup setting stream" << stream);
  cufftSetStream (plan_fwd, stream);

  DEBUG("CUDA::FilterbankEngine::setup fwd FFT plan set");

  if (freq_res > 1)
  {
    if(cufftPlan1d (&plan_bwd, freq_res, CUFFT_C2C, nchan_subband)) {
    		      throw Error (InvalidState, "dsp::FilterbankEngine:setup bad bwd_fft plan"
    		                   );
    		  }
    cufftSetStream (plan_bwd, stream);
  }

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

  if (!real_to_complex || twofft)
    return;

  DEBUG("CUDA::FilterbankEngine::setup real-to-complex");

  unsigned nfft = nchan_subband * freq_res;
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
  filterbank_cuda_perform (this, this, in);
}

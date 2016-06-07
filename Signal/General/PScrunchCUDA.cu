//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2015 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/PScrunchCUDA.h"

#include "Error.h"
#include "debug.h"

#include <memory>

#include <string.h>

using namespace std;

void check_error (const char*);

CUDA::PScrunchEngine::PScrunchEngine (cudaStream_t _stream)
{
  stream = _stream;
}

CUDA::PScrunchEngine::~PScrunchEngine ()
{
}

//! get cuda device properties
void CUDA::PScrunchEngine::setup()
{
  gpu_config.init();
}


//
// each thread reads a single value from both polarisation
// and adds them together
//
__global__ void fpt_pscrunch_kernel (float * in_p0,  float * in_p1,
                                     float * out, uint64_t in_chan_span,
                                     uint64_t out_chan_span, uint64_t in_ndat)
{
  // ichan: blockIdx.y
  const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= in_ndat)
    return;

  // increment the input/output base pointers to this chan/pol
  in_p0 += (blockIdx.y * in_chan_span);
  in_p1 += (blockIdx.y * in_chan_span);
  out   += (blockIdx.y * out_chan_span);

  out[idx] = (in_p0[idx] + in_p1[idx]) * M_SQRT1_2;
}

void CUDA::PScrunchEngine::fpt_pscrunch (const dsp::TimeSeries* input, 
                                         dsp::TimeSeries* output)
{
  if (input == output)
    throw Error (InvalidParam, "CUDA::PScrunchEngine::fpt_pscrunch",
     "cannot handle in-place data");

  const uint64_t input_ndat  = input->get_ndat();
  const unsigned input_nchan = input->get_nchan();
  const unsigned input_npol  = input->get_npol();

  if (input_npol != 2)
    throw Error (InvalidParam, "CUDA::PScrunchEngine::fpt_scrunch",
      "number of input polarisations must be two");

  uint64_t in_chan_span = 0;
  uint64_t out_chan_span = 0;
  if (input_nchan > 1)
  {
    in_chan_span = input->get_datptr (1, 0) - input->get_datptr (0, 0);
    out_chan_span = output->get_datptr (1, 0) - output->get_datptr (0, 0);
  }

  float * in_pol0 = (float *) input->get_datptr (0, 0);
  float * in_pol1 = (float *) input->get_datptr (0, 1);
  float * out     = output->get_datptr (0, 0);

#ifdef _DEBUG
  cerr << "CUDA::PScrunchEngine::fpt_pscrunch channel spans: input=" << in_chan_span << " output=" << out_chan_span << endl;
#endif

  dim3 threads (gpu_config.get_max_threads_per_block());
  dim3 blocks (input_ndat / threads.x, input_nchan);

  if (input_ndat % threads.x)
    blocks.x ++;

  // pass span as number of complex values
  fpt_pscrunch_kernel<<<blocks,threads,0,stream>>> (in_pol0, in_pol1, out, in_chan_span, out_chan_span, input_ndat); 
  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error ("CUDA::PScrunchEngine::fpt_pscrunch");
}


// each block pscrunches 1 time sample for many channels
__global__ void tfp_pscrunch_kernel (float * in, float * out, unsigned nchan)
{
  extern __shared__ float pscr_shm[];

  // isamp == blockIdx.y
  // ipol  == even/odd threads
  // ichan == blockIdx.x * blockDim.x  + threadIdx.x

  const unsigned isamp = blockIdx.y;
  const unsigned ichanpol = (blockIdx.x * blockDim.x + threadIdx.x);
  const unsigned ichan = ichanpol / 2;
  const unsigned ipol  = ichanpol & 0x1;  // % 2
  const unsigned npol  = 2;

  if (ichanpol >= nchan*npol)
    return;

  const unsigned int idx = (isamp * nchan * npol) + ichan * npol + ipol;
  const unsigned int odx = (isamp * nchan) + ichan;

  pscr_shm[threadIdx.x] = in[idx]; 

  __syncthreads();

  if (ipol == 0)
    out[odx] = (pscr_shm[threadIdx.x] + pscr_shm[threadIdx.x+1]) * M_SQRT1_2;
}

void CUDA::PScrunchEngine::tfp_pscrunch (const dsp::TimeSeries* input,
                                         dsp::TimeSeries* output)
{
  if (input == output)
    throw Error (InvalidParam, "CUDA::PScrunchEngine::tfp_pscrunch"
     "cannot handle in-place data");

  const uint64_t input_ndat  = input->get_ndat();
  const unsigned input_nchan = input->get_nchan();
  const unsigned input_npol  = input->get_npol();

  if (input_npol != 2)
    throw Error (InvalidParam, "CUDA::PScrunchEngine::fpt_scrunch",
      "number of input polarisations must be two");

  dim3 threads (gpu_config.get_max_threads_per_block());
  dim3 blocks (input_nchan*input_npol/threads.x, input_ndat);
  if (input_nchan*input_npol % threads.x)
    blocks.x++;

  float * in_base = (float *) input->get_dattfp ();
  float * out_base = (float *) output->get_dattfp ();
  size_t shm_bytes = blocks.x * sizeof(float);

  tfp_pscrunch_kernel<<<blocks,threads,shm_bytes,stream>>> (in_base, out_base, input_nchan);
}

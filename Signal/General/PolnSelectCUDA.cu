//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2015 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/PolnSelectCUDA.h"

#include "Error.h"
#include "debug.h"

#include <memory>

#include <string.h>

using namespace std;

void check_error (const char*);

CUDA::PolnSelectEngine::PolnSelectEngine (cudaStream_t _stream)
{
  stream = _stream;
}

CUDA::PolnSelectEngine::~PolnSelectEngine ()
{
}

//! get cuda device properties
void CUDA::PolnSelectEngine::setup()
{
  gpu_config.init();
}


//
// each thread reads a single value from both polarisation
// and adds them together
//
__global__ void fpt_polnselect_kernel (float * in, float * out, 
                                       uint64_t in_chan_span,
                                       uint64_t out_chan_span, 
                                       uint64_t in_ndat)
{
  // ichan: blockIdx.y
  const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= in_ndat)
    return;

  out[blockIdx.y * out_chan_span + idx] = in[blockIdx.y * in_chan_span + idx];
}

void CUDA::PolnSelectEngine::fpt_polnselect (int ipol,
                                             const dsp::TimeSeries* input, 
                                             dsp::TimeSeries* output)
{
  if (input == output)
    throw Error (InvalidParam, "CUDA::PolnSelectEngine::fpt_polnselect",
     "cannot handle in-place data");

  const uint64_t ndat  = input->get_ndat();
  const unsigned nchan = input->get_nchan();
  const unsigned npol  = input->get_npol();

  if (npol != 2)
    throw Error (InvalidParam, "CUDA::PolnSelectEngine::fpt_polnselect",
      "number of input polarisations must be two");

  uint64_t in_chan_span = 0;
  uint64_t out_chan_span = 0;
  if (nchan > 1)
  {
    in_chan_span = input->get_datptr (1, 0) - input->get_datptr (0, 0);
    out_chan_span = output->get_datptr (1, 0) - output->get_datptr (0, 0);
  }

  // TODO (idea) this could be changed to a bunch of memcpy's in low nchan case

  float * in  = (float *) input->get_datptr (0, ipol);
  float * out = output->get_datptr (0, 0);

#ifdef _DEBUG
  cerr << "CUDA::PolnSelectEngine::fpt_polnselect channel spans: input=" << in_chan_span << " output=" << out_chan_span << endl;
#endif

  dim3 threads (gpu_config.get_max_threads_per_block());
  dim3 blocks (ndat / threads.x, nchan);

  if (ndat % threads.x)
    blocks.x ++;

  // pass span as number of complex values
  fpt_polnselect_kernel<<<blocks,threads,0,stream>>> (in, out, in_chan_span, out_chan_span, ndat); 
  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error ("CUDA::PolnSelectEngine::fpt_polnselect");
}


// each block polnselectes 1 time sample for many channels
__global__ void tfp_polnselect_kernel (float * in, float * out, unsigned nchan)
{
  // isamp == blockIdx.y
  // ichan == blockIdx.x * blockDim.x  + threadIdx.x

  const unsigned isamp = blockIdx.y;
  const unsigned ichan = (blockIdx.x * blockDim.x + threadIdx.x);
  const unsigned npol  = 2;

  if (ichan >= nchan)
    return;

  const unsigned int idx = (isamp * nchan * npol) + (ichan * npol);
  const unsigned int odx = (isamp * nchan) + ichan;

  out[odx] = in[idx];
}

void CUDA::PolnSelectEngine::tfp_polnselect (int ipol,
                                             const dsp::TimeSeries* input,
                                             dsp::TimeSeries* output)
{
  if (input == output)
    throw Error (InvalidParam, "CUDA::PolnSelectEngine::tfp_polnselect"
     "cannot handle in-place data");

  const uint64_t ndat  = input->get_ndat();
  const unsigned nchan = input->get_nchan();
  const unsigned npol  = input->get_npol();

  if (npol != 2)
    throw Error (InvalidParam, "CUDA::PolnSelectEngine::fpt_scrunch",
      "number of input polarisations must be two");

  dim3 threads (gpu_config.get_max_threads_per_block());
  if (nchan < gpu_config.get_max_threads_per_block())
    threads.x = nchan;

  dim3 blocks (nchan/threads.x, ndat);
  if (nchan % threads.x)
    blocks.x++;

  // offset into the TFP array by ipol
  float * in_base = (float *) input->get_dattfp () + ipol;
  float * out_base = (float *) output->get_dattfp ();

  tfp_polnselect_kernel<<<blocks,threads,0,stream>>> (in_base, out_base, nchan);
}

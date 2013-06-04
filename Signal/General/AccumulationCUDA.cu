//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/AccumulationCUDA.h"

#include "Error.h"
#include "templates.h"
#include "debug.h"

#include <memory>

#include <string.h>

using namespace std;

void check_error (const char*);
void check_error_stream (const char*, cudaStream_t);

/*
 *  Important Note, this engine is only efficient for larger strides (256-512)
 *  stride == nbeam for molongolo
 */

CUDA::AccumulationEngine::AccumulationEngine (cudaStream_t _stream)
{
  stream = _stream;
}

// integrate a single channel / stride element (aka antenna)
__global__ void integrate_chan_beam (const float* base_in, float *base_out, unsigned tscrunch, 
                           unsigned block_stride, unsigned chan_stride)
{
  unsigned ichan    = blockIdx.y;
  unsigned ibeam    = threadIdx.x;
  unsigned idat_out  = blockIdx.x;

  unsigned in_offset = (idat_out * block_stride * tscrunch) + (ichan * chan_stride)  + ibeam;
  unsigned out_idx   = (idat_out * block_stride) + (ichan * chan_stride) + ibeam;

  float * in  = (float *) base_in + in_offset;

  __shared__ float out_val;
  out_val = 0;

  // accumulate tscrunch times
  for (unsigned i=0; i<tscrunch; i++)
  {
    out_val += in[i * block_stride];
  }

  // assign the output value
  base_out[out_idx] = out_val;
}

void CUDA::AccumulationEngine::integrate ( const dsp::TimeSeries* input, 
					 dsp::TimeSeries* output, unsigned tscrunch, unsigned stride)
{
  // nbeam == stride
  uint64_t ndat  = input->get_ndat ();
  unsigned nchan = input->get_nchan ();
  unsigned npol  = 1;    // TODO handle this case

  // distance between successive time samples for a chan / beam
  unsigned chan_stride = npol * stride;                 // size of 1 chan for 1 idat
  unsigned block_stride = nchan * chan_stride;          // size of 2D block of data

  // size of blocks for kernel
  uint64_t nblocks  = ndat / (stride * npol * tscrunch);
  dim3 threads (stride); 
  dim3 blocks (nblocks, nchan);

#ifdef _DEBUG
  cerr << "CUDA::AccumulationEngine::integrate ndat=" << ndat << endl;
  cerr << "CUDA::AccumulationEngine::integrate nchan=" << nchan << endl;
  cerr << "CUDA::AccumulationEngine::integrate chan_stride=" << chan_stride << endl;
  cerr << "CUDA::AccumulationEngine::integrate block_stride=" << block_stride << endl;
  cerr << "CUDA::AccumulationEngine::integrate nblocks=" << nblocks << endl;
  cerr << "CUDA::AccumulationEngine::integrate tscrunch=" << tscrunch << endl;
  cerr << "CUDA::AccumulationEngine::integrate stride=" << stride << endl;
#endif

  if (ndat % tscrunch)
    cerr << "CUDA::AccumulationEngine::integrate ndat mod tscrunch != 0" << endl;

  const float* base_in  = input->get_dattfp ();
  float* base_out = output->get_dattfp ();

  if (dsp::Operation::verbose)
    cerr << "CUDA::AccumulationEngine::integrate ndat=" << ndat << " tscrunch=" << tscrunch 
         << " block_stride=" << block_stride << " chan_stride=" << chan_stride << endl;

  integrate_chan_beam<<<blocks,threads,0,stream>>> (base_in, base_out, tscrunch, block_stride, chan_stride);

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    if (stream)
      check_error_stream ("CUDA::AccumulationEngine::integrate", stream);
    else
      check_error ("CUDA::AccumulationEngine::integrate");
}

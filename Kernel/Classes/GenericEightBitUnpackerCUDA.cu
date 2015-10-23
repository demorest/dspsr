//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2015 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/GenericEightBitUnpackerCUDA.h"
#include "dsp/Operation.h"

#include "Error.h"

using namespace std;

void check_error (const char*);

/*
 * Simple CUDA 8-bit unpack kernel
 * This kernel is not optimized; in particular, data access is not coalesced.
 */

__global__ void unpack (unpack_dimensions dim,
			float scale,
			const unsigned char* input,
			float* output, uint64_t output_stride)
{
  uint64_t idat = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned ichan = blockIdx.y;
  unsigned ipol = threadIdx.y;
  unsigned idim = threadIdx.z;

  unsigned input_stride = dim.nchan * dim.npol * dim.ndim;

  if (idat >= dim.ndat)
    return;

  input += input_stride*idat + dim.ndim * (dim.npol*ichan + ipol) + idim;

  output += output_stride * (dim.npol*ichan + ipol) + dim.ndim*idat + idim;

  *output = (float(*input) + 0.5) * scale;
}


void generic_8bit_unpack (cudaStream_t stream, 
			  const unpack_dimensions& dim,
			  float scale,
			  const unsigned char* input,
			  float* output, uint64_t stride)
{
  unsigned datum_threads = 256;
  if (datum_threads > dim.ndat)
    datum_threads = 32;

  unsigned datum_blocks = dim.ndat / datum_threads;
  if (dim.ndat % datum_threads)
    datum_blocks ++;

  dim3 blockDim (datum_threads, dim.npol, dim.ndim);
  dim3 gridDim (datum_blocks, dim.nchan, 1);

  unpack<<<gridDim,blockDim,0,stream>>> (dim, scale, input, output, stride);

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error ("generic_8bit_unpack");
}


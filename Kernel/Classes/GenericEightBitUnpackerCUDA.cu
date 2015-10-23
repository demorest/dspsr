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

template<typename T>
__global__ void unpack (unpack_dimensions dim,
			float scale, float offset,
			const T* input,
			float* output, uint64_t output_stride)
{
  uint64_t idat  = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned ichan = blockIdx.y;
  unsigned ipol  = threadIdx.y;
  unsigned idim  = threadIdx.z;

  unsigned input_stride = dim.nchan * dim.npol * dim.ndim;

  if (idat >= dim.ndat)
    return;

  input += input_stride*idat + dim.ndim * (dim.npol*ichan + ipol) + idim;

  output += output_stride * (dim.npol*ichan + ipol) + dim.ndim*idat + idim;

  *output = (float(*input) + offset) * scale;
}


void generic_8bit_unpack (cudaStream_t stream, 
			  const unpack_dimensions& dim,
			  const dsp::BitTable* table,
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

  if (table->get_type() == dsp::BitTable::TwosComplement)
  {
    const int8_t* signed_input = reinterpret_cast<const int8_t*> (input);
    float offset = 0.5;
    unpack<<<gridDim,blockDim,0,stream>>> (dim, table->get_scale(), offset, 
					   signed_input, output, stride);
  }
  else if (table->get_type() == dsp::BitTable::OffsetBinary)
  {
    float offset = -127.5;
    unpack<<<gridDim,blockDim,0,stream>>> (dim, table->get_scale(), offset, 
					   input, output, stride);
  }
  else
    throw Error (InvalidState, "generic_8bit_unpack",
		 "unknown BitTable::Type");

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error ("generic_8bit_unpack");
}


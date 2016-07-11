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

void check_error_stream (const char*, cudaStream_t);

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
  uint64_t idat  = blockIdx.x * blockDim.x + threadIdx.x
                 + blockIdx.z * blockDim.x * gridDim.x;
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

// defined in FoldCUDA.C
static ostream& operator << (ostream& os, const dim3& c)
{
  return os << "[" << c.x << "," << c.y << "," << c.z << "]";
}

void generic_8bit_unpack (cudaStream_t stream, 
			  const unpack_dimensions& dim,
			  const dsp::BitTable* table,
			  const unsigned char* input,
			  float* output, uint64_t stride)
{
  unsigned max_threads_per_block = 256;
  unsigned min_threads_per_block = 32;
  unsigned max_blocks_per_dim = 65535;

  unsigned datum_threads = max_threads_per_block / (dim.npol * dim.ndim);
  if (datum_threads > dim.ndat)
    datum_threads = min_threads_per_block;

  unsigned datum_blocks_x = dim.ndat / datum_threads;
  unsigned datum_blocks_z = 1;

  if (datum_blocks_x > max_blocks_per_dim)
  {
    datum_blocks_z = datum_blocks_x / max_blocks_per_dim;
    if (datum_blocks_x % max_blocks_per_dim)
      datum_blocks_z ++;

    datum_blocks_x /= datum_blocks_z;
  }

  while (dim.ndat > datum_threads * datum_blocks_z * datum_blocks_x)
    datum_blocks_x ++;

  dim3 blockDim (datum_threads, dim.npol, dim.ndim);
  dim3 gridDim (datum_blocks_x, dim.nchan, datum_blocks_z);

  // cerr << "blockDim=" << blockDim << " gridDim=" << gridDim << endl;

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
    check_error_stream ("generic_8bit_unpack", stream);
}


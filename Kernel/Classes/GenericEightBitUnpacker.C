/***************************************************************************
 *
 *   Copyright (C) 2012 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/GenericEightBitUnpacker.h"
#include "dsp/BitTable.h"

#if HAVE_CUDA
#include "dsp/MemoryCUDA.h"
#include "dsp/GenericEightBitUnpackerCUDA.h"
#include <cuda_runtime.h>
#endif

using namespace std;

static void* const undefined_stream = (void *) -1;

//! Constructor
dsp::GenericEightBitUnpacker::GenericEightBitUnpacker ()
  : EightBitUnpacker ("GenericEightBitUnpacker")
{
#define ASSUME_TWOS_COMPLEMENT 1
#if ASSUME_TWOS_COMPLEMENT
  table = new BitTable (8, BitTable::TwosComplement);
#else
  table = new BitTable (8, BitTable::OffsetBinary);
#endif
  gpu_stream = undefined_stream;
}

bool dsp::GenericEightBitUnpacker::matches (const Observation* observation)
{
  return observation->get_nbit() == 8;
}

void dsp::GenericEightBitUnpacker::unpack ()
{
#if HAVE_CUDA
  if (gpu_stream != undefined_stream)
  {
    unpack_on_gpu ();
    return;
  }
#endif

  BitUnpacker::unpack ();
}

//! Return true if the unpacker can operate on the specified device
bool dsp::GenericEightBitUnpacker::get_device_supported (Memory* memory) const
{
#if HAVE_CUDA
  if (verbose)
    cerr << "dsp::GenericEightBitUnpacker::get_device_supported HAVE_CUDA" << endl;
  return dynamic_cast< CUDA::DeviceMemory*> ( memory );
#else
  return false;
#endif
}

//! Set the device on which the unpacker will operate
void dsp::GenericEightBitUnpacker::set_device (Memory* memory)
{
#if HAVE_CUDA
  CUDA::DeviceMemory* gpu = dynamic_cast< CUDA::DeviceMemory*>( memory );
  if (gpu)
  {
    staging.set_memory( memory );
    gpu_stream = (void *) gpu->get_stream();
  }
  else
    gpu_stream = undefined_stream;

#else
  Unpacker::set_device (memory);
#endif
}

#if HAVE_CUDA

void dsp::GenericEightBitUnpacker::unpack_on_gpu ()
{
  const uint64_t ndat = input->get_ndat();

  staging.Observation::operator=( *input );
  staging.resize(ndat);

  // staging buffer on the GPU for packed data
  unsigned char* d_staging = staging.get_rawptr();
 
  const unsigned char* from= input->get_rawptr();

  cudaStream_t stream = (cudaStream_t) gpu_stream;

  cudaError error;

  if (stream)
    error = cudaMemcpyAsync (d_staging, from, input->get_nbytes(),
                             cudaMemcpyHostToDevice, stream);
  else
    error = cudaMemcpy (d_staging, from, input->get_nbytes(), 
			cudaMemcpyHostToDevice);

  if (error != cudaSuccess)
    throw Error (FailedCall, "GenericEightBitUnpacker::unpack_on_gpu",
                 "cudaMemcpy%s %s", stream?"Async":"", 
                 cudaGetErrorString (error));

  float* output_base = output->get_datptr(0,0);
  uint64_t output_stride = output->get_nfloat_span ();

  unpack_dimensions dim;
  dim.ndat = ndat;
  dim.nchan = input->get_nchan ();
  dim.npol = input->get_npol ();
  dim.ndim = input->get_ndim ();

  generic_8bit_unpack (stream, dim, table,
		       d_staging, output_base, output_stride);
}

#endif

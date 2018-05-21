//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2015 by Willem van Straten and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/MeerKATUnpackerCUDA.h"
#include "dsp/Operation.h"
#include "dsp/MemoryCUDA.h"

#include "Error.h"

#include <cuComplex.h>

using namespace std;

void check_error_stream (const char*, cudaStream_t);

// each thread unpacks 1 complex sample
__global__ void meerkat_unpack_fpt_kernel (const uint64_t ndat, float scale, const char2 * input,
cuFloatComplex * output, uint64_t ostride)
{
  // blockIdx.x is the heap number, threadIdx.x is the sample number in the heap
  const uint64_t idat = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idat >= ndat)
    return;

  const unsigned ichanpol    = blockIdx.y * gridDim.z + blockIdx.z; // ichan * npol + ipol
  const unsigned pol_stride  = gridDim.y * blockDim.x;   // nchan * heap_size
  const unsigned heap_stride = gridDim.z * pol_stride;  // npol * pol_stride

  //                    iheap                        ipol                        ichan      * heap_size
  const uint64_t idx = (blockIdx.x * heap_stride) + (blockIdx.z * pol_stride) + (blockIdx.y * blockDim.x) + threadIdx.x;
  const uint64_t odx = (ichanpol * ostride) + idat;

  char2 in16 = input[idx];

  cuFloatComplex out64;
  out64.x  = ((float) in16.x + 0.5) * scale;
  out64.y  = ((float) in16.y + 0.5) * scale;

  output[odx] = out64;
}


// each thread unpacks 1 complex sample
__global__ void meerkat_unpack_fpt_swap2_kernel (const uint64_t ndat, float scale, const char2 * input, cuFloatComplex * output, uint64_t ostride)
{
  // blockIdx.x is the heap number, threadIdx.x is the sample number in the heap
  const uint64_t idat = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idat >= ndat)
    return;

  const unsigned ichanpol    = blockIdx.y * gridDim.z + blockIdx.z; // ichan * npol + ipol
  const unsigned pol_stride  = gridDim.y * blockDim.x;   // nchan * heap_size
  const unsigned heap_stride = gridDim.z * pol_stride;  // npol * pol_stride

  //                    iheap                        ipol                        ichan      * heap_size
  const uint64_t idx = (blockIdx.x * heap_stride) + (blockIdx.z * pol_stride) + (blockIdx.y * blockDim.x) + threadIdx.x;

  // switch odd and even samples
  const int64_t roach2_fix = 1 - (2 * (idat & 0x1));

  const uint64_t odx = (ichanpol * ostride) + idat + roach2_fix;

  char2 in16 = input[idx];

  cuFloatComplex out64;
  out64.x  = ((float) in16.x + 0.5) * scale;
  out64.y  = ((float) in16.y + 0.5) * scale;

  output[odx] = out64;
}

CUDA::MeerKATUnpackerEngine::MeerKATUnpackerEngine (cudaStream_t _stream)
{
  stream = _stream;
}

void CUDA::MeerKATUnpackerEngine::setup ()
{
  // determine cuda device properties for block & grid size
  int device;
  cudaGetDevice(&device);
  cudaGetDeviceProperties (&gpu, device);
}

bool CUDA::MeerKATUnpackerEngine::get_device_supported (dsp::Memory* memory) const
{
  return dynamic_cast< CUDA::DeviceMemory*> ( memory );
}

void CUDA::MeerKATUnpackerEngine::set_device (dsp::Memory* memory)
{
  //CUDA::DeviceMemory * gpu_mem = dynamic_cast< CUDA::DeviceMemory*>( memory );
  //staging.set_memory (gpu_mem);
}


void CUDA::MeerKATUnpackerEngine::unpack (float scale, const dsp::BitSeries * input, dsp::TimeSeries * output, unsigned sample_swap)
{
  const uint64_t ndat = input->get_ndat();
  const unsigned nchan = input->get_nchan();
  const unsigned ndim = input->get_ndim();
  const unsigned npol = input->get_npol();

#ifdef _DEBUG
  cerr << "CUDA::MeerKATUnpackerEngine::unpack scale=" << scale 
       << " ndat=" << ndat << " nchan=" << nchan << " ndim=" << ndim 
       << " npol=" << npol << endl;
#endif

  // copy from CPU Bitseries to GPU staging Bitseries
  char2 * from   = (char2 *) input->get_rawptr();

  cuFloatComplex * into = (cuFloatComplex *) output->get_datptr(0, 0);
  size_t pol_span = (output->get_datptr(0, 1) - output->get_datptr(0,0)) / ndim;

  if (dsp::Operation::verbose)
    cerr << "CUDA::MeerKATUnpackerEngine::unpack from=" << (void *) from
         << " to=" << (void *) into << " pol_span=" << pol_span << endl;

  // since 256 samples per heap
  int nthread = 256;

  // each thread will unpack 4 time samples
  dim3 blocks = dim3 (ndat / nthread, nchan, npol);

  if (ndat % nthread != 0)
    blocks.x++;

#ifdef _DEBUG
  cerr << "CUDA::MeerKATUnpackerEngine::unpack meerkat_unpack ndat=" << ndat 
       << " scale=" << scale << " input=" << (void*) input << " nblock=(" 
       << blocks.x << "," << blocks.y << "," << blocks.z << ")" << " nthread=" << nthread << endl;
#endif

  if (sample_swap == 2)
    meerkat_unpack_fpt_swap2_kernel<<<blocks,nthread,0,stream>>> (ndat, scale, from, into, pol_span);
  else if (sample_swap) 
    meerkat_unpack_fpt_kernel<<<blocks,nthread,0,stream>>> (ndat, scale, from, into, pol_span);

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error_stream ("CUDA::MeerKATUnpackerEngine::unpack", stream);
}


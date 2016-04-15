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

//#define _DEBUG 

using namespace std;

void check_error (const char*);

// each thread unpacks 1 complex sample
__global__ void meerkat_unpack_fpt_kernel (const uint64_t ndat, float scale, const char2 * input, cuFloatComplex * output, uint64_t ostride)
{
  const uint64_t idat = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idat >= ndat)
    return;

  //                        ichan      * npol      + ipol
  const unsigned ichanpol = blockIdx.y * gridDim.z + blockIdx.z;

  const uint64_t idx = (ichanpol * ndat)    + idat;
  const uint64_t odx = (ichanpol * ostride) + idat;

  char2 in16 = input[idx];

  cuFloatComplex out64;
  out64.x  = ((float) in16.x + 0.5) * scale;
  out64.y  = ((float) in16.y + 0.5) * scale;

  //if (blockIdx.y == 0 && blockIdx.z == 0)
  //  printf ("in[%lu] (%d,%d) out[%lu] (%f,%f)\n", idx, in16.x, in16.y, odx, out64.x, out64.y);
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


void CUDA::MeerKATUnpackerEngine::unpack (float scale, const dsp::BitSeries * input, dsp::TimeSeries * output)
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

  int nthread = gpu.maxThreadsPerBlock;

  // each thread will unpack 4 time samples
  dim3 blocks = dim3 (ndat / nthread, nchan, npol);

  if (ndat % nthread != 0)
    blocks.x++;

#ifdef _DEBUG
  cerr << "CUDA::MeerKATUnpackerEngine::unpack meerkat_unpack ndat=" << ndat 
       << " scale=" << scale << " input=" << (void*) input << " nblock=(" 
       << blocks.x << "," << blocks.y << "," << blocks.z << ")" << " nthread=" << nthread << endl;
#endif

  meerkat_unpack_fpt_kernel<<<blocks,nthread,0,stream>>> (ndat, scale, from, into, pol_span);

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error ("CUDA::MeerKATUnpackerEngine::unpack");

  // put it here for now
  cudaStreamSynchronize(stream);
}


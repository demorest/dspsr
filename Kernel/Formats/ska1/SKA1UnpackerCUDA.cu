//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <stdio.h>
#include <cuda_runtime.h>

#include "dsp/SKA1UnpackerCUDA.h"
#include "dsp/Operation.h"
#include "dsp/MemoryCUDA.h"

#include "Error.h"
#define WARP_SIZE 32
#define BLOCK_SIZE 1024
//#define _GDEBUG

using namespace std;

void check_error_stream (const char*, cudaStream_t);

__global__ void k_unpack_fpt (float2 * to, const char2 * from,
                              uint64_t ndat, uint64_t ostride,
                              float scale)
{
  const uint64_t idat = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idat >= ndat)
    return;

  const unsigned ichanpol = blockIdx.y * gridDim.z + blockIdx.z;
  const unsigned blk_stride = gridDim.y * gridDim.z * BLOCK_SIZE;

  //                   iblk                      ichanpol              isamp
  const uint64_t idx = (blockIdx.x*blk_stride) + ichanpol*BLOCK_SIZE + threadIdx.x;
  const uint64_t odx = (ichanpol * ostride) + idat;

  char2 packed = from[idx];
  float2 unpacked;
  unpacked.x = ((float) packed.x + 0.5) * scale;
  unpacked.y = ((float) packed.y + 0.5) * scale;
  //unpacked.x = (float) scale;
  //unpacked.y = (float) scale;
  to[odx] = unpacked;
}

__global__ void k_unpack_tfp (uint64_t nval, float scale,
                              float2 * to, const int16_t * from,
                              const unsigned nchan, const unsigned npol,
                              size_t pol_span, unsigned nval_per_thread,
                              unsigned nval_per_block)
{
  extern __shared__ int16_t sdata[];

  // shared memory for this block
  const unsigned ndim = 2;
  const unsigned warp_num = threadIdx.x / WARP_SIZE;
  const unsigned warp_idx = threadIdx.x % WARP_SIZE;
  const unsigned block_offset = blockIdx.x * nval_per_block;
  unsigned idx = (warp_num * (WARP_SIZE * nval_per_thread)) + warp_idx;

  // read input data as 2 x int8_t pairs into shm
  unsigned ival;
  for (ival=0; ival<nval_per_thread; ival++)
  {
    //if (blockIdx.x == 0)
    //  printf ("[%d] sdata[%d]=from[%d]\n", threadIdx.x, idx, (block_offset + idx));

    if (idx < nval_per_block && (block_offset + idx) < nval)
      sdata[idx] = from[block_offset + idx];
    //else
    //  sdata[idx] = 0;

    idx += WARP_SIZE;
  }

  __syncthreads();

  // for use in access each 8bit value
  int8_t * sdata8 = (int8_t *) sdata;

  // determine which channel and polarisation this warp should be writing out (coalesced)
  const unsigned nchanpol = nchan * npol;
  const unsigned ichanpol_block = warp_num * nval_per_thread;
  unsigned ichunk   = ichanpol_block / nchanpol;
  unsigned ichanpol = ichanpol_block % nchanpol;
  unsigned isamp    = (ichunk * WARP_SIZE) + warp_idx;

  unsigned nsamp_per_block = nval_per_block / (nchan * ndim);
  const unsigned out_block_offset = blockIdx.x * nsamp_per_block;
  unsigned out_idx  = (pol_span * ichanpol) + out_block_offset + isamp;
  unsigned sout_idx = isamp * nchanpol + ichanpol;

  float2 val;
  for (ival=0; ival<nval_per_thread; ival++)
  {
    if (2*sout_idx < nval_per_block)
    {
      val.x = scale * float(sdata8[2*sout_idx+0]) + 0.5;
      val.y = scale * float(sdata8[2*sout_idx+1]) + 0.5;
      to[out_idx] = val;

      // increment the output ichan/pol 
      ichanpol++;
      if (ichanpol >= nchanpol)
      { 
        ichanpol = 0;
        ichunk++;
        isamp += WARP_SIZE;
        out_idx = out_block_offset + isamp;
      }
      else
        out_idx += pol_span;
    }
  }

}

#ifdef SKA1_ENGINE_IMPLEMENTATION

CUDA::SKA1UnpackerEngine::SKA1UnpackerEngine (cudaStream_t _stream)
{
  stream = _stream;
}

void CUDA::SKA1UnpackerEngine::setup ()
{
  // determine cuda device properties for block & grid size
  int device;
  cudaGetDevice(&device);
  cudaGetDeviceProperties (&gpu, device);
}

bool CUDA::SKA1UnpackerEngine::get_device_supported (dsp::Memory* memory) const
{
  return dynamic_cast< CUDA::DeviceMemory*> ( memory );
}

void CUDA::SKA1UnpackerEngine::set_device (dsp::Memory* memory)
{
  CUDA::DeviceMemory * gpu_mem = dynamic_cast< CUDA::DeviceMemory*>( memory );
  staging.set_memory( gpu_mem);
}


void CUDA::SKA1UnpackerEngine::unpack (float scale, const dsp::BitSeries * input, dsp::TimeSeries * output)
{
  const uint64_t ndat = input->get_ndat();
  const unsigned nchan = input->get_nchan();
  const unsigned ndim = input->get_ndim();
  const unsigned npol = input->get_npol();

  // gpu staging buffer for input Bitseries Block
  staging.Observation::operator=( *input );
  staging.resize(ndat);

  // copy from CPU Bitseries to GPU staging Bitseries
  void * from   = (void *) input->get_rawptr();
  void * staged = (void *) staging.get_rawptr();
  uint64_t nval = ndat * nchan * npol;
  uint64_t nbytes = nval * ndim;

  if (dsp::Operation::verbose)
    cerr << "CUDA::SKA1UnpackerEngine::unpack from=" << from
         << " to=" << staged << " nbytes=" << nbytes << endl;

  // ensure no GPU related operations are pending on this stream
  cudaStreamSynchronize (stream);

  cudaError_t error = cudaMemcpyAsync (staged, from, nbytes, cudaMemcpyHostToDevice, stream);
  if (error != cudaSuccess)
    throw Error (FailedCall, "CUDA::SKA1Unpacker::unpack",
                     "cudaMemcpyAsync %s", cudaGetErrorString (error));

  float * into    = (float *) output->internal_get_buffer();
  size_t pol_span = output->get_datptr(0, 1) - output->get_datptr(0,0);

  unsigned chunk_size = gpu.warpSize;
  unsigned nchunk_per_block = gpu.sharedMemPerBlock / (chunk_size * nchan * npol * ndim);
  unsigned nval_per_block = nchunk_per_block * chunk_size * nchan * npol;

  unsigned nthreads = gpu.maxThreadsPerBlock;
  unsigned nblocks = nval / nval_per_block;
  if (nval % nval_per_block > 0)
    nblocks++;

  unsigned nval_per_thread = nval_per_block / nthreads;
  if (nval_per_block % nthreads)
    nval_per_thread++;

  size_t sbytes = nval_per_block * ndim;

  // unpack dem bits
  k_unpack<<<nblocks,nthreads,sbytes,stream>>> (nval, scale, (float2 *) into, (int16_t *) staged, nchan, npol, pol_span, nval_per_thread, nval_per_block);

  cudaStreamSynchronize(stream);

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error_stream ("CUDA::SKA1UnpackerEngine::unpack", stream);

}
#else
void ska1_unpack_tfp (cudaStream_t stream, uint64_t nval, float scale, 
                  float * into, void * staged, 
                  unsigned  nchan, unsigned npol, unsigned ndim, 
                  size_t pol_span)
{
  const unsigned warpSize = 32;
  const unsigned sharedMemPerBlock = 49152;
  const unsigned maxThreadsPerBlock = 1024;

  unsigned chunk_size = warpSize;
  unsigned nchunk_per_block = sharedMemPerBlock / (chunk_size * nchan * npol * ndim);
  unsigned nval_per_block = nchunk_per_block * chunk_size * nchan * npol;

  unsigned nthreads = maxThreadsPerBlock;
  unsigned nblocks = nval / nval_per_block;
  if (nval % nval_per_block > 0)
    nblocks++;

  unsigned nval_per_thread = nval_per_block / nthreads;
  if (nval_per_block % nthreads)
    nval_per_thread++;

  size_t sbytes = nval_per_block * ndim;

//#ifdef _GDEBUG
  cerr << "nval=" << nval << " scale=" << scale << " nchan=" << nchan << " npol=" << npol << " pol_span=" << pol_span << endl;
  cerr << "into=" << (void *) into << " staged = " << staged << endl;
  cerr << "nblocks=" << nblocks << " nthreads=" << nthreads << " sbytes=" << sbytes << endl;
  cerr << "nval_per_thread=" << nval_per_thread << " nval_per_block=" << nval_per_block << endl;
//#endif

  // unpack dem bits
  k_unpack_tfp<<<nblocks,nthreads,sbytes,stream>>> (nval, scale, (float2 *) into, (int16_t *) staged, nchan, npol, pol_span, nval_per_thread, nval_per_block);
  
  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error_stream ("CUDA::SKA1UnpackerEngine::unpack", stream);

  return;
}

void ska1_unpack_fpt (cudaStream_t stream, uint64_t ndat, float scale,
                      float * into, void * from, unsigned nchan,
                      size_t pol_span)
{
  const unsigned nthreads = 1024;
  const unsigned npol = 2;
  const unsigned ndim = 2;

  dim3 blocks (ndat / nthreads,  nchan, npol);
  if (ndat % nthreads)
    blocks.x++;

  // output pol stride in uints of float2
  const uint64_t pol_stride = (uint64_t) pol_span / ndim;

#ifdef _GDEBUG
  cerr << "ndat=" << ndat << " nchan=" << nchan << " pol_span=" << pol_span << endl;
  cerr << "pol_stride=" << pol_stride << endl;
  cerr << "into=" << (void *) into << " from=" << from << endl;
  cerr << "nblocks=" << blocks.x << " nthreads=" << nthreads << endl;
#endif

  //uint64_t myscale = reinterpret_cast<uint64_t>(stream);
  //scale = (float) myscale;
  //cerr << "stream=" << (void *) stream << " scale=" << scale << endl;

  //const unsigned sdata_bytes = nthreads * ndim;
  k_unpack_fpt<<<blocks,nthreads,0,stream>>> ((float2 *) into, (char2 *) from, ndat, pol_stride, scale);

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error_stream ("CUDA::SKA1UnpackerEngine::k_unpack_fpt", stream);

  return;
}
#endif

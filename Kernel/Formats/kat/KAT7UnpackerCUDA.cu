//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/KAT7UnpackerCUDA.h"
#include "dsp/Operation.h"

#include "Error.h"

#include <cuComplex.h>

using namespace std;

void check_error_stream (const char*, cudaStream_t);

// each thread unpacks samples so that 1 warp does 128 contiguous samples
__global__ void kat7_unpack_fpt_kernel (const uint64_t ndat, float scale, const int16_t * input, cuFloatComplex * output)
{
  const int warp_idx = threadIdx.x & 0x1F;	// threadIDx.x % 32
  const int warp_num = threadIdx.x / 32;

  const unsigned ichan = blockIdx.y;
  const unsigned nchan = gridDim.y;
  const unsigned ichan_offset = (ichan * 128);

  // first sample for the start of the warp
  unsigned isamp = (blockIdx.x * blockDim.x * 4) + (warp_num * 128) + warp_idx;
  unsigned idx = (blockIdx.x * blockDim.x * 4 * nchan) + (warp_num * nchan * 128) + ichan_offset + warp_idx;
  unsigned odx = (ichan * ndat) + isamp;

  int16_t val16;
  int8_t * val8 = (int8_t *) &val16;
  cuFloatComplex val64;

  for (unsigned ival=0; ival<4; ival++)
  {
    if (isamp < ndat)
    {
      val16 = input[idx];
      val64.x = ((float) val8[0] + 0.5) * scale;
      val64.y = ((float) val8[1] + 0.5) * scale;
      output[odx] = val64;

      idx += 32;
      odx += 32;
      isamp += 32;
    }
  }

}

void kat7_unpack (cudaStream_t stream, const uint64_t ndat, unsigned nchan, unsigned npol,
                  float scale, const int16_t * input, float * output)
{
  int nthread = 1024;

  const unsigned ndat_per_block = 4 * nthread;

  // each thread will unpack 4 time samples
  dim3 blocks = dim3 (ndat / ndat_per_block, nchan);

  if (ndat % ndat_per_block != 0)
    blocks.x++;

#ifdef _DEBUG
  cerr << "kat7_unpack ndat=" << ndat << " scale=" << scale 
       << " input=" << (void*) input << " nblock=(" << blocks.x << "," << blocks.y << ")"
       << " nthread=" << nthread << endl;
#endif

  kat7_unpack_fpt_kernel<<<blocks,nthread,0,stream>>> (ndat, scale, input, (cuFloatComplex *) output);

  // AJ's theory... 
  // If there are no stream synchronises on the input then the CPU pinned memory load from the
  // input class might be able to get ahead of a whole sequence of GPU operations, and even exceed
  // one I/O loop. Therefore this should be a reuqirement to have a stream synchronize some time
  // after the data are loaded from pinned memory to GPU ram and the next Input copy to pinned memory

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error_stream ("kat7_unpack_fpt_kernel", stream);

  // put it here for now
  cudaStreamSynchronize(stream);

}

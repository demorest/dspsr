//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2016 by Andre Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SKComputerCUDA.h"
#include "dsp/MemoryCUDA.h"

#include "Error.h"
#include "templates.h"
#include "debug.h"

#include <stdio.h>
#include <memory>
#include <string.h>

#include <cuComplex.h>

#ifdef __CUDA_ARCH__
    #if (__CUDA_ARCH__ >= 300)
        #define HAVE_SHFL
    #else
        #define NO_SHFL
    #endif
#endif

using namespace std;

void check_error (const char*);
void check_error_stream (const char*, cudaStream_t);

/*
 *  Important Note, this engine is only efficient for larger strides (256-512)
 *  stride == nbeam for molongolo
 */

CUDA::SKComputerEngine::SKComputerEngine (dsp::Memory * memory)
{
  device_memory = dynamic_cast<CUDA::DeviceMemory *>(memory);
  stream = device_memory->get_stream();

  work_buffer_size = 0;
  work_buffer = 0;
}

void CUDA::SKComputerEngine::setup ()
{
  if (dsp::Operation::verbose)
    cerr << "CUDA::SKComputerEngine::setup ()" << endl;

  // determine GPU capabilities
  int device = 0;
  cudaGetDevice(&device);
  struct cudaDeviceProp device_properties;
  cudaGetDeviceProperties (&device_properties, device);
  max_threads_per_block = device_properties.maxThreadsPerBlock;
}

// each
__global__ void reduce_sqld_new (float2 * in, float2 * sums, float * skestimates, uint64_t in_stride, unsigned M)
{
  extern __shared__ float s1s[];
  float * s2s = s1s + 32;

  // each block integrates M samples
  const unsigned ichanpol = blockIdx.y;
  const unsigned nchanpol = gridDim.y;

  // offset to current channel, pol
  in += (ichanpol * in_stride) + (blockIdx.x * M);

  float power;
  float2 val;
  float s1 = 0;
  float s2 = 0;

  // in case M is > blockDim.x
  for (unsigned i=threadIdx.x; i<M; i+=blockDim.x)
  {
    // load the complex value 
    val = in[i];

    power = (val.x * val.x) + (val.y * val.y);
    s1 += power;
    s2 += (power * power);
  }

#ifdef HAVE_SHFL
  s1 += __shfl_down (s1, 16);
  s1 += __shfl_down (s1, 8);
  s1 += __shfl_down (s1, 4);
  s1 += __shfl_down (s1, 2);
  s1 += __shfl_down (s1, 1);

  s2 += __shfl_down (s2, 16);
  s2 += __shfl_down (s2, 8);
  s2 += __shfl_down (s2, 4);
  s2 += __shfl_down (s2, 2);
  s2 += __shfl_down (s2, 1);

  unsigned warp_idx = threadIdx.x % 32;
  unsigned warp_num = threadIdx.x / 32;
  
  if (warp_idx == 0)
  {
    s1s[warp_num] = s1;
    s2s[warp_num] = s2;
  }
  __syncthreads(); 

  if (warp_num == 0)
  {
    s1 = s1s[warp_idx];
    s2 = s2s[warp_idx];

    s1 += __shfl_down (s1, 16);
    s1 += __shfl_down (s1, 8);
    s1 += __shfl_down (s1, 4);
    s1 += __shfl_down (s1, 2);
    s1 += __shfl_down (s1, 1);

    s2 += __shfl_down (s2, 16);
    s2 += __shfl_down (s2, 8);
    s2 += __shfl_down (s2, 4);
    s2 += __shfl_down (s2, 2);
    s2 += __shfl_down (s2, 1);

    // s1 and s2 sums across block are complete
    if (warp_idx == 0)
    {
      val.x = s1;
      val.y = s2;
      unsigned odx = blockIdx.x*nchanpol + ichanpol;
      sums [odx] = val;
      skestimates[odx] = ((M+1) / (M-1)) * (M * (s2 / (s1 * s1)) - 1);
    }
  }
#endif
#ifdef NO_SHFL

  s1s[threadIdx.x] = s1;
  s2s[threadIdx.x] = s2;

  __syncthreads();

  int last_offset = blockDim.x/2;
  for (int offset = last_offset; offset > 0;  offset >>= 1)
  {
    if (threadIdx.x < offset)
    {
      s1s[threadIdx.x] += s1s[threadIdx.x + offset];
      s2s[threadIdx.x] += s2s[threadIdx.x + offset];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0)
  {
    val.x = s1s[0];
    val.y = s2s[0];
    unsigned odx = blockIdx.x*nchanpol + ichanpol;
    sums [odx] = val;
    skestimates[odx] = ((M+1) / (M-1)) * (M * (val.y / (val.x * val.x)) - 1);
  }


#endif

  // now we need to a reduction across the block
}


/* Perform a reduction including SQLD calculations */
__global__ void reduce_sqld (float * in, float * out, const uint64_t ndat)
{
  extern __shared__ float sdata[];

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int s1 = (threadIdx.x*2);
  unsigned int s2 = (threadIdx.x*2) + 1;

  float re = 0;
  float im = 0;
  if (i < ndat)
  {
    re = in[(2*i)];
    im = in[(2*i) + 1];
  }

  sdata[s1] = (re * re) + (im * im);
  sdata[s2] = sdata[s1] * sdata[s1];

  __syncthreads();

  int last_offset = blockDim.x/2 + blockDim.x % 2;

  for (int offset = blockDim.x/2; offset > 0;  offset >>= 1)
  {
    // add a partial sum upstream to our own
    if (threadIdx.x < offset)
    {
      sdata[s1] += sdata[s1 + (2*offset)];
      sdata[s2] += sdata[s2 + (2*offset)];
    }
    __syncthreads();

    // special case for non power of 2 reductions
    if ((last_offset % 2) && (last_offset > 2) && (threadIdx.x == offset))
    {
      sdata[0] += sdata[s1 + (2*offset)];
      sdata[1] += sdata[s2 + (2*offset)];
    }

    last_offset = offset;

    // wait until all threads in the block have updated their partial sums
    __syncthreads();
  }

  // thread 0 writes the final result
  if (threadIdx.x == 0)
  {
    out[(2*blockIdx.x)]   = sdata[0];
    out[(2*blockIdx.x)+1] = sdata[1];
  }
}

/* sum each set of S1 and S2 and compute SK estimate for whole block */
__global__ void reduce_sk_estimate_new (float2* input, float * output, unsigned nchanpol, unsigned ndat, float M)
{
  // input are stored in TFP order
  const float M_fac = (M+1) / (M-1);

  for (unsigned ichanpol=threadIdx.x; ichanpol<nchanpol; ichanpol+=blockDim.x)
  {
    float2* in = input;
    float2 sum = make_cuComplex(0,0);;

    for (unsigned idat=0; idat<ndat; idat++)
    {
      sum = cuCaddf (sum, in[ichanpol]);
      in += nchanpol;
    }
    output[ichanpol] = M_fac * (M * (sum.y/ (sum.x * sum.x)) - 1);
  }
}


__global__ void reduce_sk_estimate (float * in, float * out, const uint64_t ndat, float M, unsigned ichan)
{
  extern __shared__ float sdata[];

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int s1 = (threadIdx.x*2);
  unsigned int s2 = (threadIdx.x*2) + 1;

  // load input into shared memory
  float re = 0;
  float im = 0;
  if (i < ndat)
  {
    re = in[(2*i)];
    im = in[(2*i) + 1];
  }

  sdata[s1] = re;
  sdata[s2] = im;

  __syncthreads();

  int last_offset = blockDim.x/2 + blockDim.x % 2;
  for (int offset = blockDim.x/2; offset > 0;  offset >>= 1)
  {
    // add a partial sum upstream to our own
    if (threadIdx.x < offset)
    {
      sdata[s1] += sdata[s1 + (2*offset)];
      sdata[s2] += sdata[s2 + (2*offset)];
    }

    __syncthreads();

    // special case for non power of 2 reductions
    if ((last_offset % 2) && (last_offset > 2) && (threadIdx.x == offset))
    {
      sdata[0] += sdata[s1 + (2*offset)];
      sdata[1] += sdata[s2 + (2*offset)];
    }

    last_offset = offset;

    // wait until all threads in the block have updated their partial sums
    __syncthreads();
  }

  // thread 0 writes the final result
  if (threadIdx.x == 0)
  {
    if (sdata[0] == 0)
      out[0] = 0;
    else
    {
      float M_fac = (M+1) / (M-1);
      out[0] = M_fac * (M * (sdata[1] / (sdata[0]*sdata[0])) - 1);
    }
  }
}

__global__ void calc_sk_estimate (float * in, float * out, float M_fac, unsigned int M, size_t out_span)
{
  unsigned int i = threadIdx.x;
  float S1_sum = in[(2*i)];
  float S2_sum = in[(2*i)+1];
  if (S1_sum == 0)
    out[out_span*i] = 0;
  else
    out[out_span*i] = M_fac * (M * (S2_sum / (S1_sum * S1_sum)) - 1);
}

// calculate SK statistics
void CUDA::SKComputerEngine::compute (const dsp::TimeSeries* input,
           dsp::TimeSeries* output, dsp::TimeSeries *output_tscr, unsigned M)
{
  if (dsp::Operation::verbose)
    std::cerr << "CUDA::SKComputerEngine::compute()" << std::endl;

  const uint64_t ndat = output->get_ndat() * M;
  const unsigned nchan = input->get_nchan ();
  const unsigned npol  = input->get_npol ();
  const unsigned nchanpol = nchan * npol;

  if (dsp::Operation::verbose)
    std::cerr << "CUDA::SKComputerEngine::compute ndat=" << ndat << " nchan="
              << nchan << " npol=" << npol << " M=" << M << std::endl;

  float * outdat = output->get_dattfp();
  float * outdat_tscr = output_tscr->get_dattfp();
  if (dsp::Operation::verbose)
  {
    std::cerr << "CUDA::SKComputerEngine::compute outdat=" << (void *) outdat << endl;
    std::cerr << "CUDA::SKComputerEngine::compute outdat_tscr=" << (void *) outdat_tscr << endl;
  }

  // TODO: currently only support FPT on GPU due to FoldCUDA
  switch (input->get_order())
  {
    case dsp::TimeSeries::OrderFPT:
    {
      if (dsp::Operation::verbose)
        std::cerr << "CUDA::SKComputerEngine::compute OrderFPT" << std::endl;

      float2 * indat = (float2*) input->get_datptr (0, 0);

      unsigned nthreads = 1024;
      if (M < nthreads)
        nthreads = M;
      dim3 blocks (ndat / M, nchanpol);

      // this is by design, due to input buffering
      assert (ndat % M == 0);

      // work buffer for S1 and S2 values for each set of M samples 
      size_t bytes_required = nchanpol * blocks.x * sizeof(float2);
      if (bytes_required > work_buffer_size)
      {
        if (work_buffer)
        {
          cudaFree(work_buffer);
        }
        work_buffer_size = bytes_required;
        cudaMalloc (&work_buffer, work_buffer_size);
      }

      if (dsp::Operation::verbose)
        cerr << "CUDA::SKComputerEngine::compute ndat=" << ndat
             << " blocks=(" << blocks.x << "," << blocks.y << ")"
             << " nthreads=" << nthreads << endl; 

      // require an S1 and S2 value for each warp in each block
      size_t shm_bytes_1 = 32 * sizeof(float2);

      if (dsp::Operation::verbose)
        cerr << "CUDA::SKComputerEngine::compute work_buffer=" << (void *) work_buffer << endl;

      uint64_t in_stride;
      if (npol > 1)
        in_stride = input->get_datptr (0, 1) - input->get_datptr (0, 0);
      else
        in_stride = input->get_datptr (1, 0) - input->get_datptr (0, 0);

      // for float2
      in_stride /= 2;

      reduce_sqld_new<<<blocks,nthreads,shm_bytes_1,stream>>> ((float2 *) indat, (float2 *) work_buffer, outdat, in_stride, M);
      if (dsp::Operation::record_time || dsp::Operation::verbose)
        if (stream)
          check_error_stream ("CUDA::SKComputerEngine::compute reduce_sqld_new [first]", stream);
      else
        check_error ("CUDA::SKComputerEngine::compute reduce_sqld_new [first]");

      // compute a tscrunched output SK
      nthreads = 1024;
      if (nchanpol < nthreads)
        nthreads = nchanpol;
      reduce_sk_estimate_new<<<1,nthreads,0,stream>>>((float2*) work_buffer, outdat_tscr, nchanpol, blocks.x, ndat);

#if 0


      // TODO consider making ichan a ydim?
      for (unsigned ichan=0; ichan<nchan; ichan++)
      {
        for (unsigned ipol=0; ipol<npol; ipol++)
        {
          indat = const_cast<float*>(input->get_datptr (ichan, ipol));

          //cerr << "CUDA::SKComputerEngine::compute ichan=" << ichan << " pol=" << ipol << " indat=" << indat << endl;

          // foreach block reduce to S1, S2 sums [out of place]
          //cerr << "CUDA::SKComputerEngine::compute [1] [" << ichan << ", " << ipol << "] shm_bytes=" << shm_bytes_1 << endl;
          reduce_sqld<<<nblocks,block_size,shm_bytes_1, stream>>> (indat, work_buffer, ndat_proc);
          if (dsp::Operation::record_time || dsp::Operation::verbose)
            if (stream)
              check_error_stream ("CUDA::SKComputerEngine::compute reduce_sqld [first]", stream);
            else
              check_error ("CUDA::SKComputerEngine::compute reduce_sqld [first]");

          // calculate S1, S2 sums for tscr [in place]
          //cerr << "CUDA::SKComputerEngine::compute [2] [" << ichan << ", " << ipol << "] shm_bytes=" << shm_bytes_2 << endl;
          reduce_sk_estimate<<<1,nblocks,shm_bytes_2,stream>>> (work_buffer, outdat_tscr, nblocks, ndat_proc, ichan);
          if (dsp::Operation::record_time || dsp::Operation::verbose)
            if (stream)
              check_error_stream ("CUDA::SKComputerEngine::compute reduce_sqld [second]", stream);
            else
              check_error ("CUDA::SKComputerEngine::compute reduce_sqld [second]");

          // caculate SK estimator for each block in place [out of place]
          calc_sk_estimate<<<1,nblocks,0,stream>>> (work_buffer, outdat, M_fac, M, nchan*npol);
          if (dsp::Operation::record_time || dsp::Operation::verbose)
            if (stream)
              check_error_stream ("CUDA::SKComputerEngine::compute sk_estimate", stream);
            else
              check_error ("CUDA::SKComputerEngine::compute sk_estimate");

          outdat ++;
          outdat_tscr ++;
        }
      }
#endif

      // now calculate the SK limit for the tscrunched data
      break;
    }

    case dsp::TimeSeries::OrderTFP:
    {
      throw Error (InvalidState, "CUDA::SKComputerEngine::compute",
                   "OrderTFP is unsupported input order");
    }

    default:
    {
      throw Error (InvalidState, "CUDA::SKComputerEngine::compute",
                   "unsupported input order");
    }
  }
}


__global__ void copy1sample ( const float * in_base,
           float2 * out_base,
           uint64_t out_stride,
           uint64_t ndat,
           unsigned M)
{
  const unsigned idat  = blockIdx.x * blockDim.x + threadIdx.x;
  if (idat >= ndat)
    return;

  const unsigned ipol  = blockIdx.z;
  const unsigned ichan = blockIdx.y;
  const unsigned isk   = idat / M;

  const unsigned nchan = gridDim.y;
  const unsigned npol = gridDim.z;

  // forward pointer to pol0 for this chan
  out_base += (ichan * npol + ipol) * out_stride;

  // get the SK estimate (TFP order) for this sample/pol
  const float sk = in_base[isk * nchan * npol + ichan*npol + ipol];

  out_base[idat].x = sk;
  out_base[idat].y = sk;
}



void CUDA::SKComputerEngine::insertsk (const dsp::TimeSeries* input, 
                                       dsp::TimeSeries* output,
                                       unsigned M)
{ 
  // copy the SK estimates to the output timesseries
  if (dsp::Operation::verbose)
    cerr << "CUDA::SKMaskerEngine::insertsk M=" << M << endl;

  uint64_t ndat  = output->get_ndat();
  unsigned nchan = output->get_nchan();
  unsigned npol  = output->get_npol();

  // order is FPT
  const float * in_base = (float *) input->get_dattfp ();
  float2 * out_base     = (float2 *) output->get_datptr (0, 0);

  uint64_t out_stride;
  if (npol == 1)
  {
    out_stride = output->get_datptr (1, 0) - output->get_datptr (0, 0);
  }
  else
  {
    out_stride = output->get_datptr (0, 1) - output->get_datptr (0, 0);
  }

  out_stride /= 2;

  unsigned threads = max_threads_per_block;
  dim3 blocks (ndat / threads, nchan, npol);
  if (ndat % threads)
    blocks.x++;

  cerr << "CUDA::SKComputerEngine::insertsk ndat=" << ndat << " nchan=" << nchan << " npol=" << npol << endl;
  cerr << "CUDA::SKComputerEngine::insertsk out_stride=" << out_stride << endl;
  cerr << "CUDA::SKComputerEngine::insertsk blocks=(" << blocks.x << ", " << blocks.y << ") threads=" << threads << endl;

  copy1sample<<<blocks,threads,0,stream>>> (in_base, out_base, out_stride, ndat, M);

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error( "CUDA::SKComputerEngine::insertsk" );
}

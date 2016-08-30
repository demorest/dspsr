//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2016 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/


#include "dsp/SKDetectorCUDA.h"

#include <iostream>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>

#include <cuComplex.h>
//#define _DEBUG 1

// TODO consider having schan / echan in mask represented by values other than 0, 1

using namespace std;

void check_error (const char*);

CUDA::SKDetectorEngine::SKDetectorEngine (dsp::Memory * memory)
{
  device_memory = dynamic_cast<CUDA::DeviceMemory *>(memory);
  stream = device_memory->get_stream();

  estimates_host = new dsp::TimeSeries();
  zapmask_host = new dsp::BitSeries();

  pinned_memory  = new PinnedMemory ();
  estimates_host->set_memory ((dsp::Memory *) pinned_memory);
  zapmask_host->set_memory ((dsp::Memory *) pinned_memory);

  transfer_estimates = new dsp::TransferCUDA (stream);
  transfer_estimates->set_kind (cudaMemcpyDeviceToHost);
  transfer_estimates->set_output( estimates_host );

  transfer_zapmask = new dsp::TransferBitSeriesCUDA (stream);
  transfer_zapmask->set_kind (cudaMemcpyDeviceToHost);
  transfer_zapmask->set_output( zapmask_host );
}

void CUDA::SKDetectorEngine::setup ()
{
  if (dsp::Operation::verbose)
    cerr << "CUDA::SKDetectorEngine::setup ()" << endl;

  // determine GPU capabilities
  int device = 0;
  cudaGetDevice(&device);
  struct cudaDeviceProp device_properties;
  cudaGetDeviceProperties (&device_properties, device);
  max_threads_per_block = device_properties.maxThreadsPerBlock;
}


// faster kernel for npol=1
__global__ void detect_one_pol (const float * indat, unsigned char * outdat, uint64_t nval, float upper, float lower)
{
  unsigned idat  = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idat < nval)
  {
    float V = indat[idat];
    if (V < lower || V > upper)
      outdat[idat] = 1;
  }
}

__global__ void detect_two_pol (const float2 * indat, unsigned char * outdat, uint64_t nval, float upper, float lower)
{
  unsigned idat  = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idat < nval)
  {
    const float2 V = indat[idat]; 
    if (V.x < lower || V.x > upper || V.y < lower || V.y > upper)
    {
      outdat[idat] = 1;
    }
  }
}


// detect SK limits for N polarisations
__global__ void detect_one_sample (const float * indat, unsigned char * outdat, uint64_t nval, float upper, float lower, unsigned npol)
{
  unsigned idat  = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (idat < nval)
  {
    unsigned zap = 0;
    float V;

    for (int ipol=0; ipol<npol; ipol++)
    {
      V = indat[(idat * npol) + ipol];
      if (V < lower || V > upper)
      {
        zap = 1;
      }
    }
    if (zap)
      outdat[idat] = 1;
  }
}

void CUDA::SKDetectorEngine::detect_ft (const dsp::TimeSeries* input,
      dsp::BitSeries* output, float upper_thresh, float lower_thresh)
{
  if (dsp::Operation::verbose)
    cerr << "CUDA::SKDetectorEngine::detect_ft()" << endl;

  const unsigned nchan = input->get_nchan();
  const unsigned npol  = input->get_npol();
  const int64_t  ndat  = input->get_ndat();

  const float * indat    = input->get_dattfp();   // TFP
  unsigned char * outdat = output->get_datptr();  // TFP also!

  uint64_t nval   = nchan * ndat;
  uint64_t nblocks  = nval / max_threads_per_block;
  if (nval % max_threads_per_block)
    nblocks++;

  dim3 threads (max_threads_per_block);
  dim3 blocks (nblocks);

  if (dsp::Operation::verbose)
  {
    cerr << "CUDA::SKDetectorEngine::detect_ft nval=" << nval << " nblocks=" << nblocks << " max_threads_per_block=" << max_threads_per_block << endl;
    cerr << "CUDA::SKDetectorEngine::detect_ft thresholds [" << lower_thresh << " - " << upper_thresh << "]" << endl;
    cerr << "CUDA::SKDetectorEngine::detect_ft npol=" << npol << endl;
  }

  if (npol == 1)
    detect_one_pol<<<blocks,threads,npol,stream>>> (indat, outdat, nval, upper_thresh, lower_thresh);
  else if (npol == 2)
    detect_two_pol<<<blocks,threads,npol,stream>>> ((const float2 *) indat, outdat, nval, upper_thresh, lower_thresh);
  else
    detect_one_sample<<<blocks,threads,npol,stream>>> (indat, outdat, nval, upper_thresh, lower_thresh, npol);

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error( "CUDA::SKDetectorEngine::detect_ft detect_one_xxx" );

#ifdef _DEBUG
  int sum = count_mask(output);
  cerr << "CUDA::SKDetectorEngine::detect_ft sum now " << sum << endl;
#endif
}

// each block reads 1 time sample, all channels/pols
// then do a block-wide sum

// input data are stored TFP, 1 warp per time sample, 32 warps / block to sum across channels
__global__ void reduce_sum_fscr_1pol (const float * input, unsigned char * out,
                                      const unsigned nchan, float lower, float upper, 
                                      unsigned schan, unsigned echan)
{
  extern __shared__ float sdata[];

  unsigned idat = blockIdx.x;
  const float * in = input + (idat * nchan);

  float sum = 0;
  for (unsigned ichan=threadIdx.x; ichan<nchan; ichan+=blockDim.x)
  {
    if (ichan >= schan && ichan < echan)
      sum += in[ichan];
  }

  sdata[threadIdx.x] = sum;
  __syncthreads();

  // now do a block wide sum across all threads
  int last_offset = blockDim.x / 2 ;
  for (int offset = last_offset; offset > 0;  offset >>= 1)
  {
    if (threadIdx.x < offset)
      sdata[threadIdx.x] += sdata[threadIdx.x + offset];

    __syncthreads();
  }

  if (threadIdx.x == 0)
  {
    float val = sdata[0] / float((echan - schan) + 1);
    if (val < lower || val > upper)
      out[idat] = 1;
  }
}

__global__ void reduce_sum_fscr_2pol (const float2 * input, unsigned char * out,
                                      const unsigned nchan, float lower, float upper, 
                                      unsigned schan, unsigned echan)
{
  extern __shared__ float2 sdata2[];

  // idat = blockIdx.x
  const float2 * in = input + (blockIdx.x * nchan);

  float2 sum = make_cuComplex(0,0);
  for (unsigned ichan=threadIdx.x; ichan<nchan; ichan+=blockDim.x)
  {
    if (ichan >= schan && ichan < echan)
      sum = cuCaddf(sum, in[ichan]);
  }

  sdata2[threadIdx.x] = sum;
  __syncthreads();

  // now do a block wide sum across all threads
  int last_offset = blockDim.x / 2;
  for (int offset = last_offset; offset > 0;  offset >>= 1)
  {
    if (threadIdx.x < offset)
      sdata2[threadIdx.x] = cuCaddf(sdata2[threadIdx.x], sdata2[threadIdx.x + offset]);
    __syncthreads();
  }

  if (threadIdx.x == 0)
  {
    float nvalidchan = float((echan - schan) + 1);
    float p0 = sdata2[0].x / nvalidchan;
    float p1 = sdata2[0].y / nvalidchan;

    if (p0 < lower || p0 > upper || p1 < lower || p1 > upper)
      out[blockIdx.x] = 1;
  }
}


void CUDA::SKDetectorEngine::detect_fscr (const dsp::TimeSeries* input, dsp::BitSeries* output, const float lower, const float upper, unsigned schan, unsigned echan)
{
  if (dsp::Operation::verbose)
    cerr << "CUDA::SKDetectorEngine::detect_fscr()" << endl;

  const unsigned nchan = input->get_nchan();
  const unsigned npol  = input->get_npol();
  const int64_t ndat   = input->get_ndat();

  const unsigned nblocks = ndat;
  unsigned nthreads = max_threads_per_block;
  if (nchan < nthreads)
    nthreads = nchan;
  const size_t shared_bytes = nthreads * npol * sizeof(float);

  // indat is the SK estimatesestimates
  const float * indat    = input->get_dattfp();

  // outdat is the bitmask
  unsigned char * outdat = output->get_datptr();

  if (dsp::Operation::verbose)
  {
    cerr << "CUDA::SKDetectorEngine::detect_fscr nchan=" << nchan << " ndat=" << ndat << endl;
    cerr << "CUDA::SKDetectorEngine::detect_fscr nblocks=" << nblocks << " nthreads=" << nthreads << " shared_bytes=" << shared_bytes << endl;
    cerr << "CUDA::SKDetectorEngine::detect_fscr thresholds [" << lower << " - " << upper << "]" << endl;
  }

  if (npol == 1)
    reduce_sum_fscr_1pol<<<nblocks,nthreads,shared_bytes,stream>>>(indat, outdat, nchan, lower, upper, schan, echan);
  else
    reduce_sum_fscr_2pol<<<nblocks,nthreads,shared_bytes,stream>>>((float2*) indat, outdat, nchan, lower, upper, schan, echan);

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error( "CUDA::SKDetectorEngine::detect_fscr_element" );

#ifdef _DEBUG
  int sum = count_mask(output);
  cerr << "CUDA::SKDetectorEngine::detect_fscr mask_sum=" << sum << endl;
#endif

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error( "CUDA::SKDetectorEngine::detect detect_fscr" );
}

__global__ void detect_tscr_element (const float * indat, unsigned char * outdat, uint64_t nval, float upper, float lower, unsigned npol, unsigned nchan)
{
  extern __shared__ char sk_tscr[];

  unsigned int idat  = (blockIdx.x * blockDim.x + threadIdx.x);

  if (idat < nval)
  {
    const unsigned nchanpol = nchan * npol;
    const unsigned ichanpol = idat % nchanpol;

    // first nchan threads to fill shared mem with the tscr SK estimates for each chan & pol (TFP)
    if (threadIdx.x < nchanpol)
    {
      sk_tscr[threadIdx.x] = (char) (indat[threadIdx.x] > upper || indat[threadIdx.x] < lower);
    }
    __syncthreads();

    outdat[idat/npol] = sk_tscr[ichanpol];
  }
}


void CUDA::SKDetectorEngine::detect_tscr (const dsp::TimeSeries* input,
      const dsp::TimeSeries* input_tscr, dsp::BitSeries* output,
      float upper_thresh, float lower_thresh)
{
  if (dsp::Operation::verbose)
    cerr << "CUDA::SKDetectorEngine::detect_tscr()" << endl;
  const unsigned nchan   = input->get_nchan();
  const unsigned npol    = input->get_npol();
  const int64_t ndat     = output->get_ndat();

  // indat is the tscr mask [nchan vals]
  const float * indat    = input_tscr->get_dattfp();

  // outdat is the bitmask
  unsigned char * outdat = output->get_datptr();

  // this kernel is indexed on output rather than input
  const uint64_t nval = ndat * nchan;
  uint64_t nblocks  = nval / max_threads_per_block;
  if (nval % max_threads_per_block)
    nblocks++;

  dim3 threads (max_threads_per_block);
  dim3 blocks (nblocks);
  unsigned shared_bytes = nchan*npol*sizeof(char);

  if (dsp::Operation::verbose)
    cerr << "CUDA::SKDetectorEngine::detect_tscr_element ndat=" << ndat
         << " nchan=" << nchan << " nval=" << nval
         << " max_threads=" << max_threads_per_block
         << " nblocks=" << nblocks << endl;

  detect_tscr_element<<<blocks,threads,shared_bytes,stream>>>(indat, outdat, nval, upper_thresh, lower_thresh, npol, nchan);

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error( "CUDA::SKDetectorEngine::detect_tscr_element" );

#ifdef _DEBUG
  int sum = count_mask(output);
  cerr << "CUDA::SKDetectorEngine::detect_tscr mask_sum=" << sum << endl;
#endif
}


void CUDA::SKDetectorEngine::reset_mask (dsp::BitSeries* output)
{
  unsigned nchan         = output->get_nchan();
  int64_t ndat           = output->get_ndat();
  unsigned char * outdat = output->get_datptr();

  size_t nbytes = nchan * ndat;

  cudaError error = cudaMemsetAsync (outdat, 0, nbytes, stream);
  if (error != cudaSuccess)
    throw Error (FailedCall, "CUDA::SKDetectorEngine::reset_mask ",
                 "cudaMemset (%p, 0, %u): %s", outdat, nbytes,
                 cudaGetErrorString (error));
#ifdef _DEBUG
  int sum = count_mask(output);
  cerr << "CUDA::SKDetectorEngine::reset_mask sum now " << sum << endl;
#endif
}

int CUDA::SKDetectorEngine::count_mask (const dsp::BitSeries* output)
{
  unsigned char * outdat = const_cast<unsigned char *>(output->get_datptr());
  const unsigned nchan   = output->get_nchan();
  const int64_t ndat     = output->get_ndat();
  int sum = 0;
/*
  const uint64_t nval    = (uint64_t) ndat * nchan;
  cudaStreamSynchronize(stream);
  thrust::device_ptr<unsigned char> d = thrust::device_pointer_cast(outdat);
  int sum = thrust::reduce(thrust::cuda::par.on(stream), d, d+nval, (int) 0, thrust::plus<int>());
  cudaStreamSynchronize(stream);
*/

  return sum;
}

float * CUDA::SKDetectorEngine::get_estimates (const dsp::TimeSeries * input)
{
  transfer_estimates->set_input (input);
  transfer_estimates->operate ();
  cudaStreamSynchronize (stream);
  return estimates_host->get_dattfp();
}

unsigned char * CUDA::SKDetectorEngine::get_zapmask (const dsp::BitSeries * input)
{
  transfer_zapmask->set_input (input);
  transfer_zapmask->operate ();
  cudaStreamSynchronize (stream);
  return zapmask_host->get_datptr();
}


//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/FoldCUDA.h"
#include "dsp/MemoryCUDA.h"

//#define _DEBUG

#include "Error.h"
#include "debug.h"

#include <cuComplex.h>
#include <memory>

#ifdef __CUDA_ARCH__
    #if (__CUDA_ARCH__ >= 300)
        #define HAVE_SHFL
    #else
        #define NO_SHFL
    #endif
#endif

using namespace std;

CUDA::FoldEngine::FoldEngine (cudaStream_t _stream, bool _hits_on_gpu)
{
  use_set_bins = false;
  d_bin = 0;
  d_bin_size = 0;

  binplan = 0;
  binplan_size = 0;
  binplan_nbin = 0;

  stream = _stream;

  d_profiles = new dsp::PhaseSeries;
  d_profiles->set_memory( new CUDA::DeviceMemory(stream) );
  
  hits_on_gpu = _hits_on_gpu;
  if (hits_on_gpu)
    d_profiles->set_hits_memory (new CUDA::DeviceMemory(stream) );

  if (dsp::Operation::verbose)
    cerr << "CUDA::FoldEngine::FoldEngine hits_on_gpu=" << hits_on_gpu << endl;

  // no data on either the host or device
  synchronized = true;

}

CUDA::FoldEngine::~FoldEngine ()
{
  if (d_bin)
    cudaFree (d_bin);
}

void CUDA::FoldEngine::set_nbin (unsigned nbin)
{
  current_bin = folding_nbin = nbin;
  current_hits = 0;
  ndat_fold = 0;
  binplan_nbin = 0;
}

void CUDA::FoldEngine::set_ndat (uint64_t ndat, uint64_t idat_start)
{
  if (ndat > binplan_size)
  {
    if (binplan)
      cudaFreeHost (binplan);

    cudaMallocHost ((void**)&binplan, ndat * sizeof(bin));
    binplan_size = ndat;
  }
}

void CUDA::FoldEngine::set_bin (uint64_t idat, double d_ibin, 
        double bins_per_sample)
{
  unsigned ibin = unsigned (d_ibin);
  if (ibin != current_bin)
  {
    /* store the number of time samples to integrate
       in the interval that just ended */
    if (binplan_nbin)
      binplan[binplan_nbin-1].hits = current_hits;

    bin start;
    start.offset = idat;
    start.ibin = ibin;

    if (binplan_nbin >= binplan_size)
      throw Error (InvalidState, "CUDA::FoldEngine::set_bin",
                   "binplan nbin=%u >= size=%u", binplan_nbin, binplan_size);

    /* start a new interval */
    binplan[binplan_nbin] = start;

    binplan_nbin ++;
    current_bin = ibin;
    current_hits = 0;
  }

  ndat_fold ++;
  current_hits ++;
}

uint64_t CUDA::FoldEngine::get_bin_hits (int ibin){
  return 0; // Fix this
}
uint64_t CUDA::FoldEngine::set_bins (double phi, double phase_per_sample, uint64_t _ndat, uint64_t idat_start)
{
  return 0;
}
dsp::PhaseSeries* CUDA::FoldEngine::get_profiles ()
{
  return d_profiles;
}

void CUDA::FoldEngine::synch (dsp::PhaseSeries* output) try
{
  if (dsp::Operation::verbose)
    cerr << "CUDA::FoldEngine::synch this=" << this << endl;

  if (synchronized)
    return;

  if (dsp::Operation::verbose)
    cerr << "CUDA::FoldEngine::synch output=" << output << endl;

  if (!transfer)
    transfer = new dsp::TransferPhaseSeriesCUDA(stream);

  transfer->set_kind( cudaMemcpyDeviceToHost );
  transfer->set_input( d_profiles );
  transfer->set_output( output );
  transfer->set_transfer_hits( hits_on_gpu );
  transfer->operate ();

  synchronized = true;
}
catch (Error& error)
{
  throw error += "CUDA::FoldEngine::synch";
}

void CUDA::FoldEngine::send_binplan ()
{
  if (dsp::Operation::verbose)
    cerr << "CUDA::FoldEngine::send_binplan ndat=" << ndat_fold 
         << " intervals=" << binplan_nbin << endl;

  if (binplan_nbin == 0)
    return;

  if (current_hits)
    binplan[binplan_nbin-1].hits = current_hits;

  current_hits = 0;

  if (dsp::Operation::verbose)
    cerr << "CUDA::FoldEngine::send_binplan"
            " first=" << binplan[0].ibin << 
            " last=" << binplan[binplan_nbin-1].ibin <<
            " stream=" << stream << endl;

  uint64_t mem_size = binplan_nbin * sizeof(bin);

  if (binplan_nbin > d_bin_size)
  {
    if (d_bin)
      cudaFree (d_bin);

    cudaMalloc ((void**)&d_bin, mem_size);
    d_bin_size = binplan_nbin;
  }

  // copy the kernel accross
  cudaError error;

  if (stream)
    error = cudaMemcpyAsync (d_bin, binplan, mem_size,
                             cudaMemcpyHostToDevice, stream);
  else
    error = cudaMemcpy (d_bin, binplan, mem_size, cudaMemcpyHostToDevice);

  if (error != cudaSuccess)
    throw Error (InvalidState, "CUDA::FoldEngine::set_binplan",
                 "cudaMemcpy%s %s", 
                 stream?"Async":"", cudaGetErrorString (error));
}

/* All CUDA folding kernels utilise the dimensionality:
 *   ipol = blockIdx.z
 *   npol = gridDim.z
 *   ichan = blockIdx.y
 *   nchan = gridDim.y
 */

// 2dim data
__global__ void fold1bin2dim (const cuFloatComplex * in_base,
           unsigned in_span,
           cuFloatComplex * out_base,
           unsigned out_span,
           unsigned nbin,
           unsigned binplan_size,
           const CUDA::bin* binplan)
{
  in_base  += in_span  * (blockIdx.y * gridDim.z + blockIdx.z);
  out_base += out_span * (blockIdx.y * gridDim.z + blockIdx.z);

  for (unsigned ibin=threadIdx.x; ibin<binplan_size; ibin+=blockDim.x)
  {
    const cuFloatComplex * input = in_base + binplan[ibin].offset;
    cuFloatComplex total = make_cuComplex (0,0);
    for (unsigned i=0; i < binplan[ibin].hits; i++)
      total = cuCaddf (total, input[i]);
    const unsigned output_ibin = binplan[ibin].ibin;
    atomicAdd(&(out_base[output_ibin].x), total.x);
    atomicAdd(&(out_base[output_ibin].y), total.y);
  }
}

__global__ void fold1bin2dim_shared (const cuFloatComplex * in_base,
           unsigned in_span,
           cuFloatComplex * out_base,
           unsigned out_span,
           unsigned nbin,
           unsigned binplan_size,
           const CUDA::bin* binplan)
{
  // one shared memory bin for each output phase bin for this chanpol
  extern __shared__ cuFloatComplex f1b2d_shared[];

  // pointers for the current channel and polarisation
  in_base  += in_span  * (blockIdx.y * gridDim.z + blockIdx.z);
  out_base += out_span * (blockIdx.y * gridDim.z + blockIdx.z);

  // coalesced read the existing phase bin values
  for (unsigned ibin=threadIdx.x; ibin<nbin; ibin+=blockDim.x)
    f1b2d_shared[ibin] = out_base[ibin];

  __syncthreads();

  for (unsigned ibin=threadIdx.x; ibin<binplan_size; ibin+=blockDim.x)
  {
    // input pointer for this phase bin
    const cuFloatComplex * input = in_base + binplan[ibin].offset;
    cuFloatComplex total = make_cuComplex (0,0);
    for (unsigned i=0; i < binplan[ibin].hits; i++)
      total = cuCaddf (total, input[i]);
    const unsigned output_ibin = binplan[ibin].ibin;
    atomicAdd(&(f1b2d_shared[output_ibin].x), total.x);
    atomicAdd(&(f1b2d_shared[output_ibin].y), total.y);
  }

  __syncthreads();

  // coalesced write the new phase bin values
  for (unsigned ibin=threadIdx.x; ibin<nbin; ibin+=blockDim.x)
    out_base[ibin] = f1b2d_shared[ibin];
}

// each warp will fold a single binplan bin
__global__ void fold1bin2dim_warp (const float2* in_base,
           unsigned in_span,
           float2* out_base,
           unsigned out_span,
           unsigned nbin,
           unsigned binplan_size,
           CUDA::bin* binplan)
{
  extern __shared__ cuFloatComplex warp_fold[];

  const int warps_per_block = blockDim.x / 32;
  const int warp_idx = threadIdx.x & 0x1F;      // % 32
  const int warp_num = threadIdx.x / 32;
  
  // the ibin that threads in this warp will add up together
  const int ibin = blockIdx.x * warps_per_block + warp_num;

  cuFloatComplex total = make_cuComplex (0,0);

  // only add up bins that we have
  if (ibin < binplan_size)
  {
    in_base += in_span * (blockIdx.y * gridDim.z + blockIdx.z);

    // start/end sample for this input bin
    const int sbin = binplan[ibin].offset;
    const int ebin = sbin + binplan[ibin].hits;

    // each thread of a warp will load samples for this ibin
    for (int i=sbin+warp_idx; i<ebin; i+=32)
    {
      total = cuCaddf (total, in_base[i]);
    }
  }

  // now add totals together
#ifdef HAVE_SHFL
  total.x += __shfl_down (total.x, 16);
  total.x += __shfl_down (total.x, 8);
  total.x += __shfl_down (total.x, 4);
  total.x += __shfl_down (total.x, 2);
  total.x += __shfl_down (total.x, 1);

  total.y += __shfl_down (total.y, 16);
  total.y += __shfl_down (total.y, 8);
  total.y += __shfl_down (total.y, 4);
  total.y += __shfl_down (total.y, 2);
  total.y += __shfl_down (total.y, 1);

  // copy to shm for warp 0 to write out to gmem
  if (warp_idx == 0)
    warp_fold[warp_num] = total; 
  __syncthreads();

  if (warp_num == 0)
  {
    out_base += out_span * (blockIdx.y * gridDim.z + blockIdx.z);
    const int ibin = blockIdx.x * warps_per_block + warp_idx;
    if (ibin >= binplan_size)
      return;
    int output_ibin = binplan[ibin].ibin;
    out_base[ output_ibin ] = cuCaddf (out_base[ output_ibin ], warp_fold[warp_idx]);
  }
#endif
#ifdef NO_SHFL
  int last_offset = 16;
  warp_fold[threadIdx.x] = total;
  __syncthreads();
  for (int offset = last_offset; offset > 0;  offset >>= 1)
  {
    if (warp_idx < offset)
      warp_fold[threadIdx.x] = cuCaddf(warp_fold[threadIdx.x], warp_fold[threadIdx.x + offset]);
    __syncthreads();
  }

  if (warp_idx == 0)
  {
    if (ibin < binplan_size)
    {
      out_base += out_span * (blockIdx.y * gridDim.z + blockIdx.z);
      int output_ibin = binplan[ibin].ibin;
      out_base[ output_ibin ] = cuCaddf (out_base[ output_ibin ], warp_fold[threadIdx.x]);
    }
  }
#endif
}


// 1dim kernels 
__global__ void fold1bin1dim_shared (const float* in_base, unsigned in_span,
           float* out_base, unsigned out_span, 
           unsigned nbin, unsigned binplan_size,
           CUDA::bin* binplan)
{
  // one shared memory bin for each output phase bin for this chanpol
  extern __shared__ float f1b1d_shared[];

  // pointers for the current channel and polarisation
  in_base  += in_span  * (blockIdx.y * gridDim.z + blockIdx.z);
  out_base += out_span * (blockIdx.y * gridDim.z + blockIdx.z);

  // coalesced read the existing phase bin values
  for (unsigned ibin=threadIdx.x; ibin<nbin; ibin+=blockDim.x)
    f1b1d_shared[ibin] = out_base[ibin];

  __syncthreads();

  for (unsigned ibin=threadIdx.x; ibin<binplan_size; ibin+=blockDim.x)
  {
    // input pointer for this phase bin
    const float * input = in_base + binplan[ibin].offset;
    float total = 0;
    for (unsigned i=0; i < binplan[ibin].hits; i++)
      total += input[i];
    const unsigned obin = binplan[ibin].ibin;
    atomicAdd(&(f1b1d_shared[obin]), total);
  }

  __syncthreads();

  // coalesced write all the new phase bin values
  for (unsigned ibin=threadIdx.x; ibin<nbin; ibin+=blockDim.x)
    out_base[ibin] = f1b1d_shared[ibin];
}

__global__ void fold1bin1dim (const float* in_base, unsigned in_span,
           float* out_base, unsigned out_span,
           unsigned nbin, unsigned binplan_size,
           CUDA::bin* binplan)
{
  in_base  += in_span  * (blockIdx.y * gridDim.z + blockIdx.z);
  out_base += out_span * (blockIdx.y * gridDim.z + blockIdx.z);

  for (unsigned ibin=threadIdx.x; ibin<binplan_size; ibin+=blockDim.x)
  {
    const float* input = in_base + binplan[ibin].offset;
    float total = 0;
    for (unsigned i=0; i < binplan[ibin].hits; i++)
      total += input[i];
    const unsigned obin = binplan[ibin].ibin;
    atomicAdd(&(out_base[obin]), total);
  }
}

// 2dim kernel hits
__global__ void fold1bin2dimhits_shared (const float2* in_base, unsigned in_span,
           float2* out_base, unsigned out_span,
           unsigned* hits_base,
           unsigned nbin, unsigned binplan_size, CUDA::bin* binplan)
{
  // one shared memory bin for each output phase bin for this chanpol
  extern __shared__ float2 f1b2dh_shared[];
  unsigned * hits_shared = (unsigned *) (f1b2dh_shared + nbin);

  //           stride   * (  ichan    *  npol      + ipol     )
  in_base   += in_span  * (blockIdx.y * gridDim.z + blockIdx.z);
  out_base  += out_span * (blockIdx.y * gridDim.z + blockIdx.z);
  hits_base += nbin * blockIdx.y;

  // coalesced read the existing phase bin values
  for (unsigned ibin=threadIdx.x; ibin<nbin; ibin+=blockDim.x)
  {
    f1b2dh_shared[ibin] = out_base[ibin];
    if (blockIdx.z == 0)
      hits_shared[ibin] = hits_base[ibin];
  }
  __syncthreads();

  for (unsigned ibin=threadIdx.x; ibin < binplan_size; ibin += blockDim.x)
  {
    const float2* input = in_base + binplan[ibin].offset;
    float2 total = make_cuComplex (0,0);
    unsigned hits = 0;
    for (unsigned i=0; i < binplan[ibin].hits; i++)
    {
      total = cuCaddf( total, input[i]);
      if (blockIdx.z == 0)
        hits += (input[i].x != 0);
    }
    const unsigned obin = binplan[ibin].ibin;
    atomicAdd(&(f1b2dh_shared[obin].x), total.x);
    atomicAdd(&(f1b2dh_shared[obin].y), total.y);
    if (blockIdx.z == 0)
      atomicAdd(&(hits_shared[obin]), hits);
  }

  __syncthreads();

  // coalesced write all the new phase bin values
  for (unsigned ibin=threadIdx.x; ibin<nbin; ibin+=blockDim.x)
  {
    out_base[ibin] = f1b2dh_shared[ibin];
    if (blockIdx.z == 0)
      hits_base[ibin] = hits_shared[ibin];
  }
}

__global__ void fold1bin2dimhits (const float2* in_base, unsigned in_span,
           float2* out_base, unsigned out_span,
           unsigned* hits_base,
           unsigned nbin, unsigned binplan_size, CUDA::bin* binplan)
{
  //           stride   * (  ichan    *  npol      + ipol     )
  in_base   += in_span  * (blockIdx.y * gridDim.z + blockIdx.z);
  out_base  += out_span * (blockIdx.y * gridDim.z + blockIdx.z);
  hits_base += nbin * blockIdx.y;

  for (unsigned ibin=threadIdx.x; ibin < binplan_size; ibin += blockDim.x)
  {
    const float2* input = in_base + binplan[ibin].offset;
    float2 total = make_cuComplex (0,0);
    unsigned hits = 0;
    for (unsigned i=0; i < binplan[ibin].hits; i++)
    {
      total = cuCaddf( total, input[i]);
      if (blockIdx.z == 0)
        hits += (input[i].x != 0);
    }
    const unsigned obin = binplan[ibin].ibin;
    atomicAdd(&(out_base[obin].x), total.x);
    atomicAdd(&(out_base[obin].y), total.y);
    if (blockIdx.z == 0)
      atomicAdd(&(hits_base[obin]), hits);
  }
}

__global__ void fold1bin1dimhits_shared (const float* in_base, unsigned in_span,
           float* out_base, unsigned out_span,
           unsigned* hits_base,
           unsigned nbin, unsigned binplan_size, CUDA::bin* binplan)
{
  // one shared memory bin for each output phase bin for this chanpol
  extern __shared__ float f1b1dh_shared[];
  unsigned * hits_shared = (unsigned *) (f1b1dh_shared + nbin);

  //           stride   * (  ichan    *  npol      + ipol     )
  in_base   += in_span  * (blockIdx.y * gridDim.z + blockIdx.z);
  out_base  += out_span * (blockIdx.y * gridDim.z + blockIdx.z);
  hits_base += nbin * blockIdx.y;

  // coalesced read the existing phase bin values
  for (unsigned ibin=threadIdx.x; ibin<nbin; ibin+=blockDim.x)
  {
    f1b1dh_shared[ibin] = out_base[ibin];
    if (blockIdx.z == 0)
      hits_shared[ibin] = hits_base[ibin];
  }
  __syncthreads();

  for (unsigned ibin=threadIdx.x; ibin < binplan_size; ibin += blockDim.x)
  {
    const float* input = in_base + binplan[ibin].offset;
    float total = 0;
    unsigned hits = 0;
    for (unsigned i=0; i < binplan[ibin].hits; i++)
    {
      total += input[i];
      if (blockIdx.z == 0)
        hits += (input[i] != 0);
    }
    const unsigned obin = binplan[ibin].ibin;
    atomicAdd(&(f1b1dh_shared[obin]), total);
    // for ipol == 0 only
    if (blockIdx.z == 0)
      atomicAdd(&(hits_shared[obin]), hits);
  }

  __syncthreads();

  // coalesced write all the new phase bin values
  for (unsigned ibin=threadIdx.x; ibin<nbin; ibin+=blockDim.x)
  {
    out_base[ibin] = f1b1dh_shared[ibin];
    if (blockIdx.z == 0)
      hits_base[ibin] = hits_shared[ibin];
  }
}

__global__ void fold1bin1dimhits (const float* in_base, unsigned in_span,
			     float* out_base, unsigned out_span,
           unsigned* hits_base,
			     unsigned nbin, unsigned binplan_size, CUDA::bin* binplan)
{
  //           stride   * (  ichan    *  npol      + ipol     )
  in_base   += in_span  * (blockIdx.y * gridDim.z + blockIdx.z);
  out_base  += out_span * (blockIdx.y * gridDim.z + blockIdx.z);
  hits_base += nbin * blockIdx.y;
  
  for (unsigned ibin=threadIdx.x; ibin < binplan_size; ibin += blockDim.x)
  {
    const float* input = in_base + binplan[ibin].offset;
    float total = 0;
    unsigned hits = 0;
    for (unsigned i=0; i < binplan[ibin].hits; i++)
    {
      total += input[i];
      hits += (input[i] != 0);
    }
    const unsigned obin = binplan[ibin].ibin;
    atomicAdd(&(out_base[obin]), total);
    // ipol == 0 only
    if (blockIdx.z == 0)
      atomicAdd(&(hits_base[obin]), hits);
  }
}

std::ostream& operator<< (std::ostream& ostr, const dim3& v)
{
  return ostr << "(" << v.x << "," << v.y << "," << v.z << ")";
}

void check_error (const char*);
void check_error_stream (const char*, cudaStream_t);

void CUDA::FoldEngine::fold ()
{
  setup ();
  send_binplan ();

  // total number of input phase bins to be opereated (capped at folding_nbin)
  unsigned bin_dim = folding_nbin;
  if (binplan_nbin < folding_nbin)
    bin_dim = binplan_nbin;

  // number of threads in the block (capped a max TPB)
  unsigned bin_threads = bin_dim;
  if (bin_threads > 1024);
    bin_threads = 1024;

  // to ensure block coherrency
  unsigned bin_blocks = 1;

  dim3 blockDim (bin_threads, 1, 1);
  dim3 gridDim (bin_blocks, nchan, npol);

#if 0
  cerr << "bin_dim=" << bin_dim << endl;
  cerr << "blockDim=" << blockDim << endl;
  cerr << "gridDim=" << gridDim << endl;
#endif

  DEBUG("bin_dim=" << bin_dim);
  DEBUG("bin_threads=" << bin_threads << " bin_blocks=" << bin_blocks);
  DEBUG("input=" << (void *) input << " output=" << (void *) output);
  DEBUG("input span=" << input_span << " output span=" << output_span);
  DEBUG("ndim=" << ndim << " nbin=" << folding_nbin << " binplan_nbin=" << binplan_nbin);
  DEBUG("hits_on_gpu=" << hits_on_gpu << " zeroed_samples=" << zeroed_samples << " hits_nchan=" << hits_nchan);

  size_t shared_max = 32768;
  size_t shared_bytes = folding_nbin * sizeof(float) * ndim;
  if (hits_on_gpu && zeroed_samples && hits_nchan == nchan)
  {
    shared_bytes += folding_nbin * sizeof(unsigned);
    if (ndim == 2)
    {
      if (shared_bytes <= shared_max)
        fold1bin2dimhits_shared<<<gridDim,blockDim,shared_bytes,stream>>> ((float2*)input, input_span/2,
                   (float2*) output, output_span/2, hits,
                   folding_nbin, binplan_nbin, d_bin);
      else
        fold1bin2dimhits<<<gridDim,blockDim,0,stream>>> ((float2*)input, input_span/2,
                   (float2*) output, output_span/2, hits,
                   folding_nbin, binplan_nbin, d_bin);
    }
    else
    {
      if (shared_bytes <= shared_max)
        fold1bin1dimhits_shared<<<gridDim,blockDim,shared_bytes,stream>>> (input, input_span,
                   output, output_span, hits,
                   folding_nbin, binplan_nbin, d_bin);
      else
        fold1bin1dimhits<<<gridDim,blockDim,0,stream>>> (input, input_span,
                   output, output_span, hits,
                   folding_nbin, binplan_nbin, d_bin);
    }
  }
  else
  {
    if (ndim == 2)
    {
      if (shared_bytes <= shared_max)
      {
        fold1bin2dim_shared<<<gridDim,blockDim,shared_bytes,stream>>> ((cuFloatComplex *) input, input_span/2,
                 (cuFloatComplex *) output, output_span/2,
                 folding_nbin, binplan_nbin, d_bin);
      }
      else
      {
        fold1bin2dim<<<gridDim,blockDim,0,stream>>> ((cuFloatComplex *) input, input_span/2,
                   (cuFloatComplex *) output, output_span/2,
                   folding_nbin, binplan_nbin, d_bin);
      }
/*
      dim3 threads(1024, 1, 1);
      unsigned nwarps = threads.x / 32;
      dim3 blocks (binplan_nbin/nwarps, nchan, npol);
      if (binplan_nbin % nwarps)
        blocks.x++;
      size_t sbytes = threads.x * sizeof(float2);
      fold1bin2dim_warp<<<blocks,threads,sbytes,stream>>> ((cuFloatComplex *) input, input_span/2,
                 (cuFloatComplex *) output, output_span/2,
                 folding_nbin, binplan_nbin, d_bin);
*/
    }
    else
    {
      if (shared_bytes <= shared_max)
        fold1bin1dim_shared<<<gridDim,blockDim,shared_bytes,stream>>> (input, input_span,
                   output, output_span,
                   folding_nbin, binplan_nbin, d_bin);
      else 
        fold1bin1dim<<<gridDim,blockDim,0,stream>>> (input, input_span,
                   output, output_span,
                   folding_nbin, binplan_nbin, d_bin);
    }
  }

  // profile on the device is no longer synchronized with the one on the host
  synchronized = false;

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    if (stream)
      check_error_stream ("CUDA::FoldEngine::fold", stream);
    else
      check_error ("CUDA::FoldEngine::fold");
}


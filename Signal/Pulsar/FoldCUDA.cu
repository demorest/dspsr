//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

//#define _DEBUG 1

#include "dsp/FoldCUDA.h"
#include "dsp/MemoryCUDA.h"

#include "Error.h"
#include "debug.h"

#include <memory>

using namespace std;

CUDA::FoldEngine::FoldEngine (cudaStream_t _stream, bool _hits_on_gpu)
{
	use_set_bins = false;
  d_bin = 0;
  d_bin_size = 0;

  binplan = 0;
  binplan_size = 0;
  binplan_nbin = 0;

  d_profiles = new dsp::PhaseSeries;
  d_profiles->set_memory( new CUDA::DeviceMemory );
  
  hits_on_gpu = _hits_on_gpu;
  if (hits_on_gpu)
    d_profiles->set_hits_memory (new CUDA::DeviceMemory);

  if (dsp::Operation::verbose)
    cerr << "CUDA::FoldEngine::FoldEngine hits_on_gpu=" << hits_on_gpu << endl;

  // no data on either the host or device
  synchronized = true;

  stream = _stream;
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
    transfer = new dsp::TransferPhaseSeriesCUDA;

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
            " last=" << binplan[binplan_nbin-1].ibin << endl;

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


/* 
 * CUDA Folding Kernels
 *   ipol = threadIdx.y
 *   npol = blockDim.y
 *   ichan = blockIdx.y
 *   nchan = gridDim.y
 *   idim = threadIdx.z
 */

__global__ void fold1bin (const float* in_base,
           unsigned in_span,
           float* out_base,
           unsigned out_span,
           unsigned ndim,
           unsigned nbin,
           unsigned binplan_size,
           CUDA::bin* binplan)
{

  unsigned ibin = blockIdx.x * blockDim.x + threadIdx.x;

  if (ibin >= binplan_size)
    return;

  unsigned output_ibin = binplan[ibin].ibin;

  in_base  += in_span  * (blockIdx.y * blockDim.y + threadIdx.y) + threadIdx.z;
  out_base += out_span * (blockIdx.y * blockDim.y + threadIdx.y) + threadIdx.z;

  float total = 0;

  for (; ibin < binplan_size; ibin += nbin)
  {
    const float* input = in_base + binplan[ibin].offset * ndim;

    for (unsigned i=0; i < binplan[ibin].hits; i++)
      total += input[i*ndim];
  }

  out_base[ output_ibin * ndim ] += total;
}


__global__ void fold1binhits (const float* in_base,
			     unsigned in_span,
			     float* out_base,
			     unsigned out_span,
           unsigned* hits_base,
			     unsigned ndim,
			     unsigned nbin,
			     unsigned binplan_size,
			     CUDA::bin* binplan)
{
  unsigned ibin = blockIdx.x * blockDim.x + threadIdx.x;

  if (ibin >= binplan_size)
    return;

  unsigned output_ibin = binplan[ibin].ibin;

  in_base  += in_span  * (blockIdx.y * blockDim.y + threadIdx.y) + threadIdx.z;
  out_base += out_span * (blockIdx.y * blockDim.y + threadIdx.y) + threadIdx.z;
  hits_base += nbin * blockIdx.y;

  float total = 0;
  unsigned hits = 0;

  for (; ibin < binplan_size; ibin += nbin)
  {
    const float* input = in_base + binplan[ibin].offset * ndim;

    for (unsigned i=0; i < binplan[ibin].hits; i++)
    {
      total += input[i*ndim];
      hits += (input[i*ndim] != 0);
    }
  }

  out_base[ output_ibin * ndim ] += total;
  // if ipol and idim both equal 0
  if ((threadIdx.y + threadIdx.z) == 0)
    hits_base[ output_ibin ] += hits;
}

std::ostream& operator<< (std::ostream& ostr, const dim3& v)
{
  return ostr << "(" << v.x << "," << v.y << "," << v.z << ")";
}

void check_error (const char*);

void CUDA::FoldEngine::fold ()
{
  setup ();
  send_binplan ();

  unsigned bin_dim = folding_nbin;
  if (binplan_nbin < folding_nbin)
    bin_dim = binplan_nbin;

  unsigned bin_threads = 128;
  if (bin_threads > bin_dim)
    bin_threads = 32;

  unsigned bin_blocks = bin_dim / bin_threads;
  if (bin_dim % bin_threads)
    bin_blocks ++;

  dim3 blockDim (bin_threads, npol, ndim);
  dim3 gridDim (bin_blocks, nchan, 1);

#if 0
  cerr << "blockDim=" << blockDim << endl;
  cerr << "gridDim=" << gridDim << endl;
#endif

  DEBUG("input span=" << input_span << " output span=" << output_span);
  DEBUG("ndim=" << ndim << " nbin=" << folding_nbin << " binplan_nbin=" << binplan_nbin);

  if (hits_on_gpu && zeroed_samples && hits_nchan == nchan)
  {
    fold1binhits<<<gridDim,blockDim,0,stream>>> (input, input_span,
	  				   output, output_span, hits,
		  			   ndim, folding_nbin,
			  		   binplan_nbin, d_bin);

  }
  else
  {
    fold1bin<<<gridDim,blockDim,0,stream>>> (input, input_span,
               output, output_span,
               ndim, folding_nbin,
               binplan_nbin, d_bin);

  }

  // profile on the device is no longer synchronized with the one on the host
  synchronized = false;

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error ("CUDA::FoldEngine::fold");
}


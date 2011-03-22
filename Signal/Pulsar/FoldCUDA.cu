//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/FoldCUDA.h"
#include "dsp/MemoryCUDA.h"

#include "Error.h"
#include "debug.h"

#include <memory>

using namespace std;

CUDA::FoldEngine::FoldEngine (cudaStream_t _stream)
{
  d_bin = 0;
  d_bin_size = 0;

  binplan = 0;
  binplan_size = 0;
  binplan_nbin = 0;

  d_profiles = new dsp::PhaseSeries;
  d_profiles->set_memory( new CUDA::DeviceMemory );

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
    transfer = new dsp::TransferCUDA;

  transfer->set_kind( cudaMemcpyDeviceToHost );
  transfer->set_input( d_profiles );
  transfer->set_output( output );
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

  in_base += in_span * blockIdx.y + threadIdx.z;
  out_base += out_span * blockIdx.y + threadIdx.z;

  float total = 0;

  for (; ibin < binplan_size; ibin += nbin)
  {
    const float* input = in_base + binplan[ibin].offset * ndim;

    for (unsigned i=0; i < binplan[ibin].hits; i++)
      total += input[i*ndim];
  }

  out_base[ output_ibin * ndim ] += total;
}

std::ostream& operator<< (std::ostream& ostr, const dim3& v)
{
  return ostr << "(" << v.x << "," << v.y << "," << v.z << ")";
}

void CUDA::FoldEngine::fold ()
{
  setup ();
  send_binplan ();

  dim3 blockDim (128, 1, ndim);
  dim3 gridDim (folding_nbin/128, npol*nchan, 1);

#if 0
  cerr << "blockDim=" << blockDim << endl;
  cerr << "gridDim=" << gridDim << endl;
#endif

  DEBUG("input span=" << input_span << " output span=" << output_span);

  fold1bin<<<gridDim,blockDim,0,stream>>> (input, input_span,
					   output, output_span,
					   ndim, folding_nbin,
					   binplan_nbin, d_bin);

  // profile on the device is no longer synchronized with the one on the host
  synchronized = false;

  if (dsp::Operation::record_time)
  {
    cudaThreadSynchronize ();

    cudaError error = cudaGetLastError();
    if (error != cudaSuccess)
      throw Error (InvalidState, "CUDA::FoldEngine::fold", 
		   cudaGetErrorString (error));
  }
}

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

#include <memory>

using namespace std;

CUDA::FoldEngine::FoldEngine ()
{
  d_bin = 0;
  d_bin_size = 0;

  d_profiles = new dsp::PhaseSeries;
  d_profiles->set_memory( new CUDA::DeviceMemory );

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
  binplan.resize (0);
}

void CUDA::FoldEngine::set_bin (uint64_t idat, unsigned ibin)
{
  if (ibin != current_bin)
  {
    /* store the number of time samples to integrate
       in the interval that just ended */
    if (binplan.size())
      binplan.back().hits = current_hits;

    bin start;
    start.offset = idat;
    start.ibin = ibin;

    /* start a new interval */
    binplan.push_back ( start );

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
         << " intervals=" << binplan.size() << endl;

  if (binplan.size() == 0)
    return;

  if (dsp::Operation::verbose)
    cerr << "CUDA::FoldEngine::send_binplan"
            " first=" << binplan.front().ibin << 
            " last=" << binplan.back().ibin << endl;

  uint64_t mem_size = binplan.size() * sizeof(bin);

  if (binplan.size() > d_bin_size)
  {
    if (d_bin)
      cudaFree (d_bin);

    cudaMalloc ((void**)&d_bin, mem_size);
    d_bin_size = binplan.size();
  }
 
  // copy the kernel accross
  cudaError error;
  error = cudaMemcpy (d_bin, &(binplan[0]), mem_size, cudaMemcpyHostToDevice);
  if (error != cudaSuccess)
    throw Error (InvalidState, "CUDA::FoldEngine::set_binplan",
                 "this=%x %s", this, cudaGetErrorString (error));
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

  fold1bin<<<gridDim,blockDim>>> (input, input_span, output, output_span,
                                  ndim, folding_nbin, binplan.size(), d_bin);

  // profile on the device is no longer synchronized with the one on the host
  synchronized = false;

  cudaThreadSynchronize ();

  cudaError error = cudaGetLastError();
  if (error != cudaSuccess)
    throw Error (InvalidState, "CUDA::FoldEngine::fold", 
                 cudaGetErrorString (error));

}


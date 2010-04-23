//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2010
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/FoldCUDA.h"

#include "Error.h"

#include <memory>

using namespace std;

CUDA::FoldEngine::FoldEngine ()
{
  d_bin = 0;
  d_bin_size = 0;
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
    if (current_bin != folding_nbin)
      binplan.back().hits = current_hits;

    bin start;
    start.offset = idat;
    start.ibin = ibin;

    binplan.push_back ( start );

    current_bin = ibin;
    current_hits = 0;
  }
  ndat_fold ++;
  current_hits ++;
}

void CUDA::FoldEngine::send_binplan ()
{
  if (dsp::Operation::verbose)
    cerr << "CUDA::FoldEngine::send_binplan ndat=" << ndat_fold 
         << " intervals=" << binplan.size() << endl;

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


// header for if we decide to calculate weights on gpu too...
//__global__ void calculateWeight ()
//{
//  unsigned threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
//}


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

  in_base += in_span * blockIdx.y + threadIdx.z;
  out_base += out_span * blockIdx.y + threadIdx.z;

  float total = 0;

  for (unsigned jbin=ibin; jbin < binplan_size; jbin += nbin)
  {
    const float* input = in_base + binplan[jbin].offset * ndim;

    for (unsigned i=0; i < binplan[jbin].hits; i++)
      total += input[i*ndim];
  }

  out_base[ binplan[ibin].ibin * ndim ] += total;
}

std::ostream& operator<< (std::ostream& ostr, const dim3& v)
{
  return ostr << "(" << v.x << "," << v.y << "," << v.z << ")";
}

void CUDA::FoldEngine::fold ()
{
  send_binplan ();

  dim3 blockDim (128, 1, ndim);
  dim3 gridDim (folding_nbin/128, npol*nchan, 1);

#if 0
  cerr << "blockDim=" << blockDim << endl;
  cerr << "gridDim=" << gridDim << endl;
#endif

  fold1bin<<<gridDim,blockDim>>> (input, input_span, output, output_span,
                                  ndim, folding_nbin, binplan.size(), d_bin);

  cudaThreadSynchronize ();

  cudaError error = cudaGetLastError();
  if (error != cudaSuccess)
    throw Error (InvalidState, "CUDA::FoldEngine::fold", 
                 cudaGetErrorString (error));

}


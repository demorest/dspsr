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
  binplan_ptr = 0;
  binplan_size = 0;
}

CUDA::FoldEngine::~FoldEngine ()
{
  if (binplan_ptr)
    cudaFree (binplan_ptr);
}

void CUDA::FoldEngine::set_binplan (uint64_t ndat, unsigned* bins)
{
  uint64_t mem_size = ndat * sizeof(unsigned);

  if (ndat > binplan_size)
  {
    if (binplan_ptr)
      cudaFree (binplan_ptr);

    cudaMalloc ((void**)&binplan_ptr, mem_size);
    binplan_size = ndat;
  }
 
  // copy the kernel accross
  cudaError error;
  error = cudaMemcpy (binplan_ptr, bins, mem_size, cudaMemcpyHostToDevice);
  if (error != cudaSuccess)
    throw Error (InvalidState, "CUDA::FoldEngine::set_binplan",
                 "this=%x %s", this, cudaGetErrorString (error));

  ndat_fold = ndat;
}


// header for if we decide to calculate weights on gpu too...
//__global__ void calculateWeight ()
//{
//  unsigned threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
//}


__global__ void performFold (const float* in_base,
			     unsigned in_span,
			     float* out_base,
			     unsigned out_span,
			     unsigned npol,
			     unsigned ndim,
			     unsigned ndat_fold,
			     unsigned* binplan)
{
  unsigned ichan = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned ipol = threadIdx.y;
  unsigned idim = threadIdx.z;

  in_base += in_span * ( ipol + ichan*npol ) + idim;
  out_base += out_span * ( ipol + ichan*npol ) + idim;

  float total = 0;
  unsigned ibin = binplan[0];

  for (unsigned i=0; i < ndat_fold; i++)
  {
    if (binplan[i] != ibin)
    {
      out_base[ibin*ndim] += total;
      total = 0;
      ibin = binplan[i];
    }
    total += in_base[i*ndim];
  }
}

std::ostream& operator<< (std::ostream& ostr, const dim3& v)
{
  return ostr << "(" << v.x << "," << v.y << "," << v.z << ")";
}

void CUDA::FoldEngine::fold ()
{
  dim3 blockDim (64, npol, ndim);
  dim3 gridDim (nchan/blockDim.x, 1, 1);

#if 0
  cerr << "blockDim=" << blockDim << endl;
  cerr << "gridDim=" << gridDim << endl;
#endif

  performFold<<<gridDim,blockDim>>> (input, input_span,
				     output, output_span,
				     npol, ndim,
				     ndat_fold, binplan_ptr);

  cudaThreadSynchronize ();

  cudaError error = cudaGetLastError();
  if (error != cudaSuccess)
    throw Error (InvalidState, "CUDA::FoldEngine::fold", 
                 cudaGetErrorString (error));

}


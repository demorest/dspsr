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

CUDA::FoldEngine::set_binplan (uint64_t ndat, unsigned* bins)
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
  cudaMemcpy (binplan_ptr, bins, mem_size, cudaMemcpyHostToDevice);

  ndat_fold = ndat;
}


// header for if we decide to calculate weights on gpu too...
//__global__ void calculateWeight ()
//{
//  unsigned threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
//}


__global__ void performFold (float* in_base,
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

  for (unsigned i=0; i < ndat_fold; i++)
    out_base[binplan[i]*ndim] += in_base[i*ndim];
}

void CUDA::FoldEngine::fold ()
{
  dim3 blockDim (256, npol, ndim);
  dim3 gridDim (nchan/256, 1, 1);

  performFold<<<gridDim,blockDim>>> (input, input_span,
				     output, output_span,
				     npol, ndim,
				     ndat_fold, binplan_ptr);
}






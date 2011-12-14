//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2011 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

//#define _DEBUG 1

#include "dsp/CyclicFoldEngineCUDA.h"
#include "dsp/MemoryCUDA.h"

#include "Error.h"
#include "debug.h"

#include <memory>

using namespace std;

CUDA::CyclicFoldEngineCUDA::CyclicFoldEngineCUDA (cudaStream_t _stream)
{
  d_binplan[0] = d_binplan[1] = NULL;
  d_lagdata = NULL;

  // no data on either the host or device
  synchronized = true;

  stream = _stream;
}

CUDA::CyclicFoldEngineCUDA::~CyclicFoldEngineCUDA ()
{
  // Free device mem
  if (d_binplan[0]) cudaFree(d_binplan[0]);
  if (d_binplan[1]) cudaFree(d_binplan[1]);
  if (d_lagdata) cudaFree(d_binplan[1]);
}

void CUDA::CyclicFoldEngineCUDA::synch (dsp::PhaseSeries *out) try
{

  if (dsp::Operation::verbose)
    cerr << "CUDA::CyclicFoldEngineCUDA::synch this=" << this << endl;

  if (synchronized)
    return;

  if (dsp::Operation::verbose)
    cerr << "CUDA::CyclicFoldEngineCUDA::synch output=" << output << endl;

  // TODO transfer lag data from GPU

  // Call usual synch() to do transform
  dsp::CyclicFoldEngine::synch(out);

  synchronized = true;
}
catch (Error& error)
{
  throw error += "CUDA::CyclicFoldEngineCUDA::synch";
}

void CUDA::CyclicFoldEngineCUDA::set_ndat (uint64_t _ndat, 
    uint64_t _idat_start)
{

  uint64_t old_lagdata_size = lagdata_size;

  dsp::CyclicFoldEngine::set_ndat(_ndat, _idat_start);

  if (lagdata_size > old_lagdata_size)
  {
    if (d_lagdata) cudaFree(d_lagdata);
    cudaMalloc((void**)&d_lagdata, lagdata_size * sizeof(float));
    cudaMemset(d_lagdata, 0, lagdata_size * sizeof(float));
  }

}

void CUDA::CyclicFoldEngineCUDA::zero ()
{
  dsp::CyclicFoldEngine::zero();
  if (d_lagdata && lagdata_size>0) 
    cudaMemset(d_lagdata, 0, lagdata_size * sizeof(float));
}

void CUDA::CyclicFoldEngineCUDA::send_binplan ()
{
  if (dsp::Operation::verbose)
    cerr << "CUDA::CyclicFoldEngineCUDA::send_binplan ndat=" << ndat_fold 
         << endl;

  uint64_t mem_size = binplan_size * sizeof(unsigned);

  cudaError error;

  for (unsigned i=0; i<2; i++) 
  {

    if (d_binplan[i] == NULL) 
      cudaMalloc ((void**)&d_binplan[i], mem_size);

    if (stream)
      error = cudaMemcpyAsync (d_binplan[i], binplan[i], mem_size,
                               cudaMemcpyHostToDevice, stream);
    else
      error = cudaMemcpy (d_binplan[i], binplan[i], mem_size, 
          cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
      throw Error (InvalidState, "CUDA::CyclicFoldEngineCUDA::set_binplan",
                   "cudaMemcpy%s %s", 
                   stream?"Async":"", cudaGetErrorString (error));
  }

}

void CUDA::CyclicFoldEngineCUDA::get_lagdata ()
{
  size_t lagdata_bytes = lagdata_size * sizeof(float);
  cudaError error;
  if (stream) 
    error = cudaMemcpyAsync (lagdata, d_lagdata, lagdata_bytes,
        cudaMemcpyDeviceToHost, stream);
  else
    error = cudaMemcpy (lagdata, d_lagdata, lagdata_bytes,
        cudaMemcpyDeviceToHost);

  if (error != cudaSuccess)
    throw Error (InvalidState, "CUDA::CyclicFoldEngineCUDA::get_lagdata",
                 "cudaMemcpy%s %s", 
                 stream?"Async":"", cudaGetErrorString (error));
}

/* 
 * TODO: CUDA Kernels
 *
 */

void check_error (const char*);

void CUDA::CyclicFoldEngineCUDA::fold ()
{

  // TODO state/etc checks

  setup ();
  send_binplan ();

  // TODO: call fold kernels

  // profile on the device is no longer synchronized with the one on the host
  synchronized = false;

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error ("CUDA::CyclicFoldEngineCUDA::fold");
}


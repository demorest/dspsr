/***************************************************************************
 *
 *   Copyright (C) 2015 by Matthew Kerr
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/FZoomCUDA.h"
#include "dsp/MemoryCUDA.h"

#include "Error.h"

#include <iostream>
#include <assert.h>

using namespace std;

CUDA::FZoomEngine::FZoomEngine(cudaStream_t _stream)
{
  stream = _stream;
  direction = DeviceToDevice;
}

cudaMemcpyKind CUDA::FZoomEngine::get_kind ()
{
  cudaMemcpyKind kind;
  switch ( direction )
  {
    case dsp::FZoom::Engine::DeviceToHost:
    {
      kind = cudaMemcpyDeviceToHost;
      break;
    }

    case dsp::FZoom::Engine::DeviceToDevice:
    {
      kind = cudaMemcpyDeviceToDevice;
      break;
    }

    default:
    {
      throw Error (InvalidState, "CUDA::FZoomEngine::fpt_copy",
          "invalid memory copy direction");
    }
  }
}

void CUDA::FZoomEngine::fpt_copy (
    const dsp::TimeSeries * input, dsp::TimeSeries * output,
    unsigned chan_lo, unsigned chan_hi)
{

  cudaMemcpyKind kind = cudaMemcpyDeviceToDevice;
  if ( output -> get_memory() -> on_host () )
    kind = cudaMemcpyDeviceToHost;

  if (stream)
    cudaStreamSynchronize(stream);
  else
    cudaThreadSynchronize();

  // if the strides are equal, can do a single copy
  // otherwise, the internal buffers have different reserves and must do
  // a series of copies
  uint64_t istride = input->get_datptr(1,0)-input->get_datptr(0,0);
  uint64_t ostride = output->get_datptr(1,0)-output->get_datptr(0,0);
  unsigned nchan = chan_hi - chan_lo + 1;
  bool aligned = istride == ostride;
  uint64_t to_copy = sizeof(float);
  unsigned max_chan = nchan;
  unsigned max_pol = input-> get_npol();
  if (aligned)
  {
    to_copy *= istride;
    max_chan = max_pol = 1;
  }
  else
  {
    to_copy *= input->get_ndim() * input->get_ndat();
  }

  if (dsp::Operation::verbose)
  {
    cerr << "CUDA::FZoomEngine::fpt_copy"
         << " chan_lo=" << chan_lo << " chan_hi=" << chan_hi
         << " input=" << input->get_datptr(chan_lo,0)
         << " input_ndat=" << input->get_ndat()
         << " input_ndim=" << input->get_ndim()
         << " input_npol=" << input->get_npol()
         << " istride=" << istride << endl
         << " output=" << output->get_datptr(0,0)
         << " output_ndat=" << output->get_ndat()
         << " output_ndim=" << output->get_ndim()
         << " output_npol=" << output->get_npol()
         << " ostride=" << ostride 
         << " aligned copy=" << aligned
         << " output on host="<<output->get_memory()->on_host()
         << endl;
  }


  for (unsigned i=0; i < max_chan; ++i)
  {
    for (unsigned j=0; j < max_pol; ++j)
    {
      cudaError error;
      if (stream)
        error = cudaMemcpyAsync (output->get_datptr(i,j),
                                 input->get_datptr(chan_lo+i,j),
                                 to_copy,
                                 kind,
                                 stream);
      else
        error = cudaMemcpy (output->get_datptr(i,j),
                                 input->get_datptr(i,j),
                                 to_copy,
                                 kind);

      if (error != cudaSuccess)
        throw Error (InvalidState, "CUDA::FZoomCUDA::fpt_copy",
                     cudaGetErrorString (error));

    }
  }

  // TODO -- see if this is necessary -- I think so if to host, not sure
  // about device
  cudaDeviceSynchronize();

}


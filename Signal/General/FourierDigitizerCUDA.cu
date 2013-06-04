//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2012 by Willem van Straten and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#define __FOURIER_DIGITIZER_TPB 512

#include "dsp/FourierDigitizerCUDA.h"
#include "Error.h"

#include <memory>

using namespace std;

void check_error (const char*);

__global__ void pack8bit (float scale, const float * in, int8_t * out)
{
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

  // TODO include scale
  out[i]   = (int8_t) (scale * in[i]);     // Re
  out[i+1] = (int8_t) (scale * in[i+1]);   // Im

}

CUDA::FourierDigitizerEngine::FourierDigitizerEngine (cudaStream_t _stream)
{
  stream = _stream;
}

void CUDA::FourierDigitizerEngine::pack (int nbit, const dsp::TimeSeries* input,
    dsp::BitSeries* output)
{
  const uint64_t ndat = output->get_ndat ();
  const unsigned nchan = output->get_nchan ();
  const unsigned ndim = output->get_ndim ();
  const unsigned npol = output->get_npol ();

  if (nbit != 8)
    throw Error (InvalidParam, "CUDA::FourierDigitizerEngine::pack",
		 "cannot handle nbit=%u != 8", nbit);

  //if (dsp::Operation::verbose)
    cerr << "CUDA::FourierDigitizerEngine::pack nbit=" << nbit
         << " ndat=" << ndat << " nchan=" << nchan << endl;

  uint64_t in_step = nchan * npol * ndim;
  uint64_t out_step = nchan * npol * ndim;

  //const float * in_ptr = const_cast<float*>(input->get_dattpf ());
  const float * in_ptr = const_cast<float*>(input->get_dattfp ());
  int8_t * out_ptr = (int8_t *) output->get_datptr ();
  float scale = 1.0;

  for (unsigned idat=0; idat < ndat; idat++)
  {
    for (unsigned ipol=0; ipol < npol; ipol++)
    {
      dim3 threads (512);
      dim3 blocks (nchan / 512);

      pack8bit<<<blocks,threads,0,stream>>> (scale, in_ptr, out_ptr);

      in_ptr += in_step;
      out_ptr+= out_step;
    }
  }
}

void CUDA::FourierDigitizerEngine::finish ()
{
  if (dsp::Operation::verbose)
    cerr << "CUDA::FourierDigitizerEngine::finish()" << endl;
  check_error ("CUDA::FourierDigitizerEngine::finish");
}

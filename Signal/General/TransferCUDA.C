/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/TransferCUDA.h"

#include "Error.h"

#include <iostream>
using namespace std;

//! Default constructor- always inplace
dsp::TransferCUDA::TransferCUDA(cudaStream_t _stream)
  : Transformation<TimeSeries,TimeSeries> ("CUDA::Transfer", outofplace)
{
  stream = _stream;
  input_stream = _stream;
  kind = cudaMemcpyHostToDevice;
  event = 0;
}

//! Do stuff
void dsp::TransferCUDA::transformation ()
{
  prepare ();

  if (stream)
  {
    cudaStreamSynchronize(stream);
    cudaStreamSynchronize(input_stream);
  }
  else
    cudaThreadSynchronize();

  if (verbose)
  {
    cerr << "dsp::TransferCUDA::transformation input ndat="
         << input->get_ndat() << " ndim=" << input->get_ndim();
    if (input->get_npol() > 1)
      cerr << " span=" << input->get_datptr (0,1) - input->get_datptr(0,0);
    cerr << " offset=" << input->get_datptr(0,0) - (float*)input->internal_get_buffer() << endl;
  }

  cudaError error;
  if (input_stream)
  {
    error = cudaMemcpyAsync (output->internal_get_buffer(),
                             input->internal_get_buffer(),
                             input->internal_get_size(),
                             kind,
                             input_stream);
    if (event)
      cudaEventRecord(event, input_stream);
  }
  else
    error = cudaMemcpy (output->internal_get_buffer(),
                             input->internal_get_buffer(),
                             input->internal_get_size(), kind);
  if (error != cudaSuccess)
    throw Error (InvalidState, "dsp::TransferCUDA::transformation",
                 cudaGetErrorString (error));

  if (verbose)
  {
    cerr << "dsp::TransferCUDA::transformation output ndat="
       << output->get_ndat() << " ndim=" << output->get_ndim();
    if (output->get_npol() > 1)
      cerr << " span=" << output->get_datptr (0, 1) - output->get_datptr(0,0);

    cerr << " offset=" << output->get_datptr(0,0) - (float*)output->internal_get_buffer() << endl;
  }
}

void dsp::TransferCUDA::prepare ()
{
  output->set_match( const_cast<TimeSeries*>(input.get()) );
  output->internal_match( input );
  output->copy_configuration( input );
}

void dsp::TransferCUDA::set_input_stream (cudaStream_t _input_stream, cudaEvent_t _event)
{
  input_stream = _input_stream;
  event = _event;
}

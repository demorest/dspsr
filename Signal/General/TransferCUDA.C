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

  unsigned ichan_start = output->get_ichan_start();
  
  const float* input_start = input->get_datptr(ichan_start, 0);
  const unsigned input_nfloat = input->get_datptr(input->get_nchan()/nbundle, 0) - input->get_datptr(0,0);
  
  if (verbose)
  {
    cerr << "dsp::TransferCUDA::transformation input ndat="
         << input->get_ndat() << " ndim=" << input->get_ndim();
    if (input->get_npol() > 1)
      cerr << " span=" << input->get_datptr (ichan_start,1) - input->get_datptr(ichan_start,0);
    cerr << " offset=" << input->get_datptr(ichan_start,0) - (float*)input->internal_get_buffer() << endl;
  }

  cudaError error;
  if (input_stream)
  {
    error = cudaMemcpyAsync (output->internal_get_buffer(),
                             input_start,
                             input_nfloat*sizeof(float),
                             kind,
                             input_stream);
    if (event)
      cudaEventRecord(event, input_stream);
  }
  else
    error = cudaMemcpy (output->internal_get_buffer(),
                        input_start,
                        input_nfloat*sizeof(float),
                        kind);
  if (error != cudaSuccess)
    throw Error (InvalidState, "dsp::TransferCUDA::transformation",
                 cudaGetErrorString (error));

  if (verbose)
  {
    cerr << "dsp::TransferCUDA::transformation output ndat="
       << output->get_ndat() << " ndim=" << output->get_ndim();
    if (output->get_npol() > 1)
      cerr << " span=" << output->get_datptr (ichan_start, 1) - output->get_datptr(ichan_start,0);

    cerr << " offset=" << output->get_datptr(ichan_start,0) - (float*)output->internal_get_buffer() << endl;
  }
}

void dsp::TransferCUDA::prepare ()
{
  output->set_match( const_cast<TimeSeries*>(input.get()) );
  output->internal_match( input );
  output->copy_configuration( input );
  output->set_total_nbundle( nbundle );
  output->set_input_bundle( input_bundle );
}

void dsp::TransferCUDA::set_input_stream (cudaStream_t _input_stream, cudaEvent_t _event)
{
  input_stream = _input_stream;
  event = _event;
}
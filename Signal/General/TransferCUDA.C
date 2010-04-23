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
dsp::TransferCUDA::TransferCUDA()
  : Transformation<TimeSeries,TimeSeries> ("CUDA::Transfer",outofplace,true)
{
  kind = cudaMemcpyHostToDevice;
}

//! Do stuff
void dsp::TransferCUDA::transformation ()
{
  prepare ();

  cudaThreadSynchronize();

  if (verbose)
    cerr << "dsp::TransferCUDA::transformation input ndat="
         << input->get_ndat() << " ndim=" << input->get_ndim()
         << " span=" << input->get_datptr (0, 1) - input->get_datptr(0,0)
         << " offset=" << input->get_datptr(0,0) - (float*)input->internal_get_buffer()
         << endl;

  cudaError error;
  error = cudaMemcpy (output->internal_get_buffer(), 
                      input->internal_get_buffer(), 
	                    input->internal_get_size(), kind);
  if (error != cudaSuccess)
    throw Error (InvalidState, "dsp::TransferCUDA::transformation",
                 cudaGetErrorString (error));

  if (verbose)
    cerr << "dsp::TransferCUDA::transformation output ndat=" 
       << output->get_ndat() << " ndim=" << output->get_ndim() 
       << " span=" << output->get_datptr (0, 1) - output->get_datptr(0,0)
       << " offset=" << output->get_datptr(0,0) - (float*)output->internal_get_buffer()
       << endl;
}

void dsp::TransferCUDA::prepare ()
{
  output->internal_match( input );
  output->copy_configuration( input );
}


/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/TransferBitSeriesCUDA.h"

#include "Error.h"

#include <iostream>
using namespace std;

//! Default constructor- always inplace
dsp::TransferBitSeriesCUDA::TransferBitSeriesCUDA()
  : Transformation<BitSeries,BitSeries> ("CUDA::Transfer", outofplace)
{
  kind = cudaMemcpyHostToDevice;
}

//! Do stuff
void dsp::TransferBitSeriesCUDA::transformation ()
{
  prepare ();

  cudaThreadSynchronize();

  if (verbose)
    cerr << "dsp::TransferBitSeriesCUDA::transformation input ndat="
         << input->get_ndat() << " ndim=" << input->get_ndim()
         << endl;

  cudaError error;
  error = cudaMemcpy (output->get_rawptr(), 
                      input->get_rawptr(), 
                      input->get_size(), kind);
  if (error != cudaSuccess)
    throw Error (InvalidState, "dsp::TransferBitSeriesCUDA::transformation",
                 cudaGetErrorString (error));

  if (verbose)
    cerr << "dsp::TransferBitSeriesCUDA::transformation output ndat=" 
       << output->get_ndat() << " ndim=" << output->get_ndim() 
       << endl;
}

void dsp::TransferBitSeriesCUDA::prepare ()
{
  output->internal_match( input );
  output->copy_configuration( input );
}


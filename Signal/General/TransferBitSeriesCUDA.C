/***************************************************************************
 *
 *   Copyright (C) 2011 by Andrew Jameson & Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/TransferBitSeriesCUDA.h"

#include "Error.h"

#include <assert.h>
#include <iostream>
using namespace std;

//! Default constructor- always inplace
dsp::TransferBitSeriesCUDA::TransferBitSeriesCUDA()
  : Transformation<BitSeries,BitSeries> ("CUDA::TransferBitSeries", outofplace)
{
  kind = cudaMemcpyHostToDevice;
}

//! Do stuff
void dsp::TransferBitSeriesCUDA::transformation ()
{
  prepare ();
  cudaThreadSynchronize();

  if (verbose)
    cerr << "dsp::TransferBitSeriesCUDA::transformation"
         << " out=" << (void*)output->get_rawptr()
         << " size=" << output->get_size()
         << " in=" << (void*)input->get_rawptr()
         << " size=" << input->get_size() << endl;

  cudaError error;

  assert (output->get_rawptr() != 0);
  assert (output->get_size() >= input->get_size());

  error = cudaMemcpy (output->get_rawptr(), 
                      input->get_rawptr(), 
                      input->get_size(), kind);

  if (error != cudaSuccess)
    throw Error (InvalidState, "dsp::TransferBitSeriesCUDA::transformation",
                 cudaGetErrorString (error));
}

void dsp::TransferBitSeriesCUDA::prepare ()
{
  output->internal_match( input );
  output->copy_configuration( input );
}


/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/TransferPhaseSeriesCUDA.h"

#include "Error.h"

#include <iostream>
using namespace std;

//! Default constructor- always inplace
dsp::TransferPhaseSeriesCUDA::TransferPhaseSeriesCUDA()
  : Transformation<PhaseSeries,PhaseSeries> ("CUDA::PhaseSeriesTransfer", outofplace)
{
  kind = cudaMemcpyHostToDevice;
  transfer_hits = false;
}

//! Do stuff
void dsp::TransferPhaseSeriesCUDA::transformation ()
{
  prepare ();

  cudaThreadSynchronize();

  if (verbose)
    cerr << "dsp::TransferPhaseSeriesCUDA::transformation input ndat="
         << input->get_ndat() << " ndim=" << input->get_ndim()
         << " span=" << input->get_datptr (0, 1) - input->get_datptr(0,0)
         << " offset=" << input->get_datptr(0,0) - (float*)input->internal_get_buffer()
         << endl;

  cudaError error;
  error = cudaMemcpy (output->internal_get_buffer(), 
                      input->internal_get_buffer(), 
	                    input->internal_get_size(), kind);
  if (error != cudaSuccess)
    throw Error (InvalidState, "dsp::TransferPhaseSeriesCUDA::transformation buffer",
                 cudaGetErrorString (error));

  if (verbose)
    cerr << "dsp::TransferPhaseSeriesCUDA::transformation output ndat=" 
       << output->get_ndat() << " ndim=" << output->get_ndim() 
       << " span=" << output->get_datptr (0, 1) - output->get_datptr(0,0)
       << " offset=" << output->get_datptr(0,0) - (float*)output->internal_get_buffer()
       << endl;

  if (transfer_hits)
  {
    if (verbose)
      cerr << "dsp::TransferPhaseSeriesCUDA::transformation hits_size=" 
           << input->get_hits_size() << endl;
         
    error = cudaMemcpy (output->get_hits(),
                        input->get_hits(),
                        input->get_hits_size(), kind);

    if (error != cudaSuccess)
      throw Error (InvalidState, "dsp::TransferPhaseSeriesCUDA::transformation hits",
                   cudaGetErrorString (error));
  }
}

void dsp::TransferPhaseSeriesCUDA::prepare ()
{
  output->internal_match( input );
  output->copy_configuration( input );
}


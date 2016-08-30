//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2016 by Andre Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SpectralKurtosisCUDA.h"

using namespace std;

CUDA::SpectralKurtosisEngine::SpectralKurtosisEngine (dsp::Memory * memory)
{
  work_buffer_size = 0;
  work_buffer = 0;
  
  device_memory = dynamic_cast<CUDA::DeviceMemory*>(memory);
  stream = device_memory->get_stream ();

  // sub-engines
  computer = new CUDA::SKComputerEngine (memory);
  detector = new CUDA::SKDetectorEngine (memory);
  masker   = new CUDA::SKMaskerEngine (memory);
}

void CUDA::SpectralKurtosisEngine::setup ()
{
  if (dsp::Operation::verbose)
    cerr << "CUDA::SpectralKurtosisEngine::setup ()" << endl;

  // determine GPU capabilities
  int device = 0;
  cudaGetDevice(&device);
  struct cudaDeviceProp device_properties;
  cudaGetDeviceProperties (&device_properties, device);
  max_threads_per_block = device_properties.maxThreadsPerBlock;

  computer->setup ();
  detector->setup ();
  masker->setup ();
}

void CUDA::SpectralKurtosisEngine::compute ( const dsp::TimeSeries* input,
           dsp::TimeSeries* output, dsp::TimeSeries *output_tscr, unsigned tscrunch)
{
  computer->compute (input, output, output_tscr, tscrunch);
}

void CUDA::SpectralKurtosisEngine::detect_ft (const dsp::TimeSeries* input,
      dsp::BitSeries* output, float upper_thresh, float lower_thresh)
{
  detector->detect_ft (input, output, upper_thresh, lower_thresh);
}

void CUDA::SpectralKurtosisEngine::detect_fscr (const dsp::TimeSeries* input, 
                                                dsp::BitSeries* output, 
                                                const float lower, const float upper,
                                                unsigned schan, unsigned echan)

{
  detector->detect_fscr(input, output, upper, lower, schan, echan);
}

void CUDA::SpectralKurtosisEngine::detect_tscr (const dsp::TimeSeries* input,
      const dsp::TimeSeries* input_tscr, dsp::BitSeries* output,
      float upper_thresh, float lower_thresh)
{
  detector->detect_tscr( input, input_tscr, output, upper_thresh, lower_thresh);
}

void CUDA::SpectralKurtosisEngine::reset_mask (dsp::BitSeries* output)
{
  detector->reset_mask(output);
}

int CUDA::SpectralKurtosisEngine::count_mask (const dsp::BitSeries* output)
{
  int nzapped = detector->count_mask (output);
  return nzapped;
}

float * CUDA::SpectralKurtosisEngine::get_estimates (const dsp::TimeSeries* estimates_device)
{ 
  return detector->get_estimates (estimates_device);
} 

unsigned char * CUDA::SpectralKurtosisEngine::get_zapmask (const dsp::BitSeries* zapmask_device)
{ 
  return detector->get_zapmask (zapmask_device);
} 

void CUDA::SpectralKurtosisEngine::mask (dsp::BitSeries* mask, const dsp::TimeSeries * input,
           dsp::TimeSeries * output, unsigned M)
{
  masker->perform (mask, input, output, M);
}

void CUDA::SpectralKurtosisEngine::insertsk (const dsp::TimeSeries* input, dsp::TimeSeries* out, unsigned M)
{
  computer->insertsk (input, out, M);
}


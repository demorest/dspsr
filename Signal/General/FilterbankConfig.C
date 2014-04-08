/***************************************************************************
 *
 *   Copyright (C) 2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/FilterbankConfig.h"
#include "dsp/Scratch.h"

#if HAVE_CUDA
#include "dsp/FilterbankCUDA.h"
#include "dsp/MemoryCUDA.h"
#endif

#include <iostream>
using namespace std;

using dsp::Filterbank;

Filterbank::Config::Config ()
{
  memory = 0;
  stream = 0; //(void*)-1;

  nchan = 1;
  freq_res = 0;  // unspecified
  when = After;  // not good, but the original default
}

std::ostream& dsp::operator << (std::ostream& os,
				const Filterbank::Config& config)
{
  os << config.get_nchan();
  if (config.get_convolve_when() == Filterbank::Config::Before)
    os << ":B";
  else if (config.get_convolve_when() == Filterbank::Config::During)
    os << ":D";
  else if (config.get_freq_res() != 1)
    os << ":" << config.get_freq_res();

  return os;
}

//! Extraction operator
std::istream& dsp::operator >> (std::istream& is, Filterbank::Config& config)
{
  unsigned value;
  is >> value;

  config.set_nchan (value);
  config.set_convolve_when (Filterbank::Config::After);

  if (is.eof())
    return is;

  if (is.peek() != ':')
  {
    is.fail();
    return is;
  }

  // throw away the colon
  is.get();

  if (is.peek() == 'D' || is.peek() == 'd')
  {
    is.get();  // throw away the D
    config.set_convolve_when (Filterbank::Config::During);
  }
  else if (is.peek() == 'B' || is.peek() == 'b')
  {
    is.get();  // throw away the B
    config.set_convolve_when (Filterbank::Config::Before);
  }
  else
  {
    unsigned nfft;
    is >> nfft;
    config.set_freq_res (nfft);
  }

  return is;
}

//! Set the device on which the unpacker will operate
void dsp::Filterbank::Config::set_device (Memory* mem)
{
  memory = mem;
}

void dsp::Filterbank::Config::set_stream (void* ptr)
{
  stream = ptr;
}

//! Return a new Filterbank instance and configure it
dsp::Filterbank* dsp::Filterbank::Config::create ()
{
  Reference::To<Filterbank> filterbank = new Filterbank;

  filterbank->set_nchan( get_nchan() );

  if (freq_res)
    filterbank->set_frequency_resolution ( freq_res );

#if HAVE_CUDA

  CUDA::DeviceMemory* device_memory = 
    dynamic_cast< CUDA::DeviceMemory*> ( memory );

  if ( device_memory )
  {
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>( stream );

    filterbank->set_engine (new CUDA::FilterbankEngine (cuda_stream));

    Scratch* gpu_scratch = new Scratch;
    gpu_scratch->set_memory (device_memory);
    filterbank->set_scratch (gpu_scratch);
  }

#endif

  return filterbank.release();
}


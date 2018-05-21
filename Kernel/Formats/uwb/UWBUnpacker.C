//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2014
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/UWBUnpacker.h"
#include "dsp/BitTable.h"

#include "Error.h"

#if HAVE_CUDA
#include "dsp/MemoryCUDA.h"
#include "dsp/UWBUnpackerCUDA.h"
#include <cuda_runtime.h>
#endif

#include <errno.h>

using namespace std;

static void* const undefined_stream = (void *) -1;

dsp::UWBUnpacker::UWBUnpacker (const char* _name) : HistUnpacker (_name)
{
  if (verbose)
    cerr << "dsp::UWBUnpacker ctor" << endl;
 
  set_ndig (2); 
  set_nstate (65536);
  first_block = true;

  npol = 0;
  ndim = 2;
}

dsp::UWBUnpacker::~UWBUnpacker ()
{
}

unsigned dsp::UWBUnpacker::get_output_offset (unsigned idig) const
{
  return idig % 2;
}

unsigned dsp::UWBUnpacker::get_output_ipol (unsigned idig) const
{
  return idig / 2;
}

unsigned dsp::UWBUnpacker::get_output_ichan (unsigned idig) const
{
  return idig / 2;
}

unsigned dsp::UWBUnpacker::get_ndim_per_digitizer () const
{
  return 1;
}

dsp::UWBUnpacker * dsp::UWBUnpacker::clone () const
{
  return new UWBUnpacker (*this);
}

//! Return true if the unpacker support the specified output order
bool dsp::UWBUnpacker::get_order_supported (TimeSeries::Order order) const
{
  return (order == TimeSeries::OrderFPT);
}

//! Set the order of the dimensions in the output TimeSeries
void dsp::UWBUnpacker::set_output_order (TimeSeries::Order order)
{
  output_order = order;
}

void dsp::UWBUnpacker::set_engine (Engine* _engine)
{
  if (verbose)
    cerr << "dsp::UWBUnpacker::set_engine()" << endl;
  engine = _engine;
}

//! Return true if the unpacker can operate on the specified device
bool dsp::UWBUnpacker::get_device_supported (Memory* memory) const
{
  if (verbose)
    cerr << "dsp::UWBUnpacker::get_device_supported()" << endl;
#if HAVE_CUDA
  CUDA::DeviceMemory * gpu_mem = dynamic_cast< CUDA::DeviceMemory*>( memory );
  if (gpu_mem)
  {
    CUDA::UWBUnpackerEngine * tmp = new CUDA::UWBUnpackerEngine(0);
    return tmp->get_device_supported (memory);
  }
  else
#endif
  {
    return false;
  }
}

//! Set the device on which the unpacker will operate
void dsp::UWBUnpacker::set_device (Memory* memory)
{
  if (verbose)
    cerr << "dsp::UWBUnpacker::set_device" << endl;
#if HAVE_CUDA
  CUDA::DeviceMemory * gpu_mem = dynamic_cast< CUDA::DeviceMemory*>( memory );
  if (gpu_mem)
  {
    cudaStream_t stream = gpu_mem->get_stream();
    if (verbose)
      cerr << "dsp::UWBUnpacker::set_device Device Memory supported with stream " << (void *) stream << endl;
    set_engine (new CUDA::UWBUnpackerEngine(stream));
  }
#endif
  if (engine)
  {
    if (verbose)
      cerr << "dsp::UWBUnpacker::set_device engine->set_device()" << endl;
    engine->set_device(memory);
    if (verbose)
      cerr << "dsp::UWBUnpacker::set_device engine->setup()" << endl;
    engine->setup();
  }
  else
    Unpacker::set_device (memory);
    if (verbose)
      cerr << "dsp::UWBUnpacker::set_device device prepared" << endl;
  device_prepared = true;
}

bool dsp::UWBUnpacker::matches (const Observation* observation)
{
  return observation->get_machine()== "UWB"
    && observation->get_nchan() == 1
    && observation->get_ndim() == 2
    && (observation->get_npol() == 2 || observation->get_npol() == 1)
    && observation->get_nbit() == 16;
}

void dsp::UWBUnpacker::unpack ()
{
  if (verbose)
    cerr << "dsp::UWBUnpacker::unpack()" << endl;

  npol = input->get_npol();
  set_ndig (npol*2);

  unsigned long * hist;
  for (unsigned ipol=0; ipol<npol; ipol++)
  {
    hist = get_histogram (ipol*2 + 0);
    hist = get_histogram (ipol*2 + 1);
  }

  if (engine)
  {
    if (verbose)
      cerr << "dsp::UWBUnpacker::unpack using Engine" << endl;
    engine->unpack(input, output);
    return;
  }

  // some programs (digifil) do not call set_device
  if ( ! device_prepared )
    set_device ( Memory::get_manager ());

  // Data will be stored in blocks of 2048 samples
  // that are each stored in FPT order
  const unsigned nsamp_block = 2048;
  const uint64_t ndat   = input->get_ndat();
  const uint64_t nblock = ndat / nsamp_block;

  const unsigned into_stride = nsamp_block * ndim;
  const unsigned from_pol_stride = into_stride;
  const unsigned from_stride = from_pol_stride * npol;

  const unsigned ichan = 0;
  unsigned long * hists[2];

  for (unsigned ipol=0; ipol<npol; ipol++)
  {
    hists[0] = get_histogram (ipol*2 + 0);
    hists[1] = get_histogram (ipol*2 + 1);

    // packed input pointer
    int16_t * from = (int16_t *) input->get_rawptr() + (ipol * from_pol_stride);
    // unpacked output pointer
    float * into = output->get_datptr (ichan, ipol);

    for (uint64_t iblock=0; iblock<nblock; iblock++)
    {
      for (unsigned isamp=0; isamp<nsamp_block*ndim; isamp+=2)
      {
        int16_t re = from[isamp+0]^0x8000;
        into[isamp+0] = float (re);

        int16_t im = from[isamp+1]^0x8000;
        into[isamp+1] = float (im);


        hists[0][int32_t(re)+32768]++;
        hists[1][int32_t(im)+32768]++;
      }

      into += into_stride;
      from += from_stride;
    }
  } // for each polarisation
  
  first_block = false;
}

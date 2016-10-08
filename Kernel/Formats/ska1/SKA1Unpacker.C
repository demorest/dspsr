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

#include "dsp/SKA1Unpacker.h"
#include "dsp/BitTable.h"

#include "Error.h"

#if HAVE_CUDA
#include "dsp/MemoryCUDA.h"
#include "dsp/SKA1UnpackerCUDA.h"
#include <cuda_runtime.h>
#endif

#include <errno.h>

using namespace std;

static void* const undefined_stream = (void *) -1;

dsp::SKA1Unpacker::SKA1Unpacker (const char* _name) : HistUnpacker (_name)
{
  if (verbose)
    cerr << "dsp::SKA1Unpacker ctor" << endl;

  set_nstate (256);
  table = new BitTable (8, BitTable::TwosComplement);
 
  npol = 2;
  ndim = 2;
}

dsp::SKA1Unpacker::~SKA1Unpacker ()
{
}

dsp::SKA1Unpacker * dsp::SKA1Unpacker::clone () const
{
  return new SKA1Unpacker (*this);
}

#ifdef SKA1_ENGINE_IMPLEMENTATION
void dsp::SKA1Unpacker::set_engine (Engine* _engine)
{
  engine = _engine;
}
#endif

//! Return true if the unpacker can operate on the specified device
bool dsp::SKA1Unpacker::get_device_supported (Memory* memory) const
{
  if (verbose)
    cerr << "dsp::SKA1Unpacker::get_device_supported" << endl;
#ifdef SKA1_ENGINE_IMPLEMENTATION
  if (engine)
    return engine->get_device_supported (memory);
  else
    return false;
#else
#if HAVE_CUDA
  if (verbose)
    cerr << "dsp::SKA1Unpacker::get_device_supported HAVE_CUDA" << endl;
  return dynamic_cast< CUDA::DeviceMemory*> ( memory );
#else
  return false;
#endif

#endif
}

//! Set the device on which the unpacker will operate
void dsp::SKA1Unpacker::set_device (Memory* memory)
{
  if (verbose)
    cerr << "dsp::SKA1Unpacker::set_device" << endl;
#ifdef SKA1_ENGINE_IMPLEMENTATION
  if (engine)
    engine->set_device(memory);
  else
    Unpacker::set_device (memory);
#else
#if HAVE_CUDA
  CUDA::DeviceMemory * gpu_mem = dynamic_cast< CUDA::DeviceMemory*>( memory );
  if (gpu_mem)
  {
    //cerr << "dsp::SKA1Unpacker::set_device activating GPU" << endl;
    gpu_stream = (void *) gpu_mem->get_stream();
    //staging.set_memory( gpu_mem );
  }
  else
    gpu_stream = undefined_stream;
#else
  Unpacker::set_device (memory);
#endif
#endif
  device_prepared = true;
}

bool dsp::SKA1Unpacker::matches (const Observation* observation)
{
  return observation->get_machine()== "SKA1"
    && observation->get_ndim() == 2
    && observation->get_npol() == 2
    && observation->get_nbit() == 8;
}

void dsp::SKA1Unpacker::unpack ()
{
  if (verbose)
    cerr << "dsp::SKA1Unpacker::unpack()" << endl;

#ifdef SKA1_ENGINE_IMPLEMENTATION
  if (engine)
  {
    if (verbose)
      cerr << "dsp::SKA1Unpacker::unpack using Engine" << endl;
    engine->unpack(table->get_scale(), input, output);
    return;
  }
#else
#if HAVE_CUDA
  if (gpu_stream != undefined_stream)
  {
    unpack_on_gpu ();
    return;
  }
#endif
#endif

  // some programs (digifil) do not call set_device
  if ( ! device_prepared )
    set_device ( Memory::get_manager ());

  // Data format is TFP

  const uint64_t ndat   = input->get_ndat();
  const unsigned nchan  = input->get_nchan();

  unsigned in_offset         = 0;
  const unsigned into_stride = ndim;
  const unsigned from_stride = nchan * ndim * npol;
  const float * lookup = table->get_values ();

  for (unsigned ichan=0; ichan<nchan; ichan++)
  {
    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      float * into = output->get_datptr (ichan, ipol);
      const unsigned char * from = input->get_rawptr() + in_offset;

      for (uint64_t idat=0; idat<ndat; idat++)
      {
        into[0] = lookup[ from[0] ];
        into[1] = lookup[ from[1] ];
        into += into_stride;
        from += from_stride;
      }
      in_offset += ndim;
    }
  }
}

#ifndef SKA1_ENGINE_IMPLEMENTATION
#if HAVE_CUDA
void dsp::SKA1Unpacker::unpack_on_gpu ()
{
  const uint64_t ndat = input->get_ndat();
  const unsigned nchan = input->get_nchan();
  const unsigned ndim = input->get_ndim();
  const unsigned npol = input->get_npol();

  if (ndat == 0)
    return;

  void * from = (void *) input->get_rawptr();
  cudaStream_t stream = (cudaStream_t) gpu_stream;

  uint64_t nval = ndat * nchan * npol;

  float * into    = (float *) output->get_datptr(0,0);
  size_t pol_span = output->get_datptr(0, 1) - output->get_datptr(0,0);

  ska1_unpack_fpt (stream, ndat, table->get_scale(), into, from, nchan, pol_span);
}
#endif
#endif

//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2015 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/KAT7Unpacker.h"
#include "dsp/BitTable.h"

#include "Error.h"

#if HAVE_CUDA
#include "dsp/MemoryCUDA.h"
#include "dsp/KAT7UnpackerCUDA.h"
#include <cuda_runtime.h>
#endif

#include <errno.h>

using namespace std;

static void* const undefined_stream = (void *) -1;

dsp::KAT7Unpacker::KAT7Unpacker (const char* _name) : HistUnpacker (_name)
{
  if (verbose)
    cerr << "dsp::KAT7Unpacker ctor" << endl;

  set_nstate (256);
  gpu_stream = undefined_stream;

  table = new BitTable (8, BitTable::TwosComplement);

  device_prepared = false;
}

dsp::KAT7Unpacker::~KAT7Unpacker ()
{
}

dsp::KAT7Unpacker * dsp::KAT7Unpacker::clone () const
{
  return new KAT7Unpacker (*this);
}

//! Return true if the unpacker can operate on the specified device
bool dsp::KAT7Unpacker::get_device_supported (Memory* memory) const
{
#if HAVE_CUDA
  if (verbose)
    cerr << "dsp::KAT7Unpacker::get_device_supported HAVE_CUDA" << endl;
  return dynamic_cast< CUDA::DeviceMemory*> ( memory );
#else
  return false;
#endif
}

//! Set the device on which the unpacker will operate
void dsp::KAT7Unpacker::set_device (Memory* memory)
{
#if HAVE_CUDA
  CUDA::DeviceMemory * gpu_mem = dynamic_cast< CUDA::DeviceMemory*>( memory );
  if (gpu_mem)
  {
    gpu_stream = (void *) gpu_mem->get_stream();
    if (verbose)
      cerr << "dsp::KAT7Unpacker::set_device using gpu memory" << endl;
    staging.set_memory( memory );
  }
  else
  {
    if (verbose)
      cerr << "dsp::KAT7Unpacker::set_device using cpu memory" << endl;
    gpu_stream = undefined_stream;
  }
#else
  Unpacker::set_device (memory);
#endif
  device_prepared = true;
}


bool dsp::KAT7Unpacker::matches (const Observation* observation)
{
  return observation->get_machine() == "KPSR"
    && observation->get_ndim() == 2
    && observation->get_nbit() == 8;
}

void dsp::KAT7Unpacker::unpack ()
{
#if HAVE_CUDA
  if (gpu_stream != undefined_stream)
  {
    unpack_on_gpu ();
    return;
  }
#endif

  // some programs (digifil) do not call set_device
  if ( ! device_prepared )
    set_device ( Memory::get_manager ());

  const uint64_t ndat  = input->get_ndat();
  const int8_t * from = (int8_t *) input->get_rawptr();
  float * into;
  const unsigned nchan = input->get_nchan();
  const unsigned npol = 1;

  const float* lookup = table->get_values ();

  // data is stored as 128 sample blocks of FT ordered data
  const uint64_t nblocks = ndat / 128;

  // cerr << "dsp::KAT7Unpacker::unpack ndat="<<ndat << " nchan=" << nchan << " nblocks=" << nblocks << endl;

  for (uint64_t iblock=0; iblock<nblocks; iblock++)
  {
    for (unsigned ichan=0; ichan<nchan; ichan++)
    {
      for (unsigned ipol=0; ipol<npol; ipol++)
      {
        into = output->get_datptr (ichan, ipol) + (iblock*256);
        for (unsigned isamp=0; isamp<256; isamp++)
        {
          into[isamp] = (float) from[isamp];
        }
        from += 256;
      }
    }
  }
}

unsigned dsp::KAT7Unpacker::get_resolution () const { return 128; }

#if HAVE_CUDA

void dsp::KAT7Unpacker::unpack_on_gpu ()
{
  const uint64_t ndat = input->get_ndat();
  const unsigned nchan = input->get_nchan();
  const unsigned ndim  = input->get_ndim();
  const unsigned npol  = input->get_npol();

  const uint64_t to_copy = ndat * nchan * ndim * npol;

  staging.Observation::operator=( *input );
  staging.resize(ndat);

  // staging buffer on the GPU for packed data
  unsigned char * d_staging = staging.get_rawptr();
  const unsigned char* from = input->get_rawptr();
  float * into = output->get_datptr(0,0);

  cudaStream_t stream = (cudaStream_t) gpu_stream;
  cudaError error;

  if (stream)
    error = cudaMemcpyAsync (d_staging, from, to_copy,
                             cudaMemcpyHostToDevice, stream);
  else
    error = cudaMemcpy (d_staging, from, to_copy, cudaMemcpyHostToDevice);

  if (error != cudaSuccess)
    throw Error (FailedCall, "KAT7Unpacker::unpack_on_gpu",
                 "cudaMemcpy%s %s", stream?"Async":"", 
                 cudaGetErrorString (error));

  kat7_unpack (stream, ndat, nchan, npol, table->get_scale(), (int16_t *) d_staging, into);
}

#endif


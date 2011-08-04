//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2009
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/CASPSRUnpacker.h"
#include "dsp/BitTable.h"

#include "Error.h"

#if HAVE_CUDA
#include "dsp/MemoryCUDA.h"
#include "dsp/CASPSRUnpackerCUDA.h"
#include <cuda_runtime.h>
#endif

#define FAST_UNPACK

using namespace std;

static void* const undefined_stream = (void *) -1;

dsp::CASPSRUnpacker::CASPSRUnpacker (const char* _name) : HistUnpacker (_name)
{
  if (verbose)
    cerr << "dsp::CASPSRUnpacker ctor" << endl;

  set_nstate (256);
  gpu_stream = undefined_stream;

  table = new BitTable (8, BitTable::TwosComplement);
}

dsp::CASPSRUnpacker::~CASPSRUnpacker ()
{
}

dsp::CASPSRUnpacker * dsp::CASPSRUnpacker::clone () const
{
  return new CASPSRUnpacker (*this);
}

//! Return true if the unpacker can operate on the specified device
bool dsp::CASPSRUnpacker::get_device_supported (Memory* memory) const
{
#if HAVE_CUDA
  if (verbose)
    cerr << "dsp::CASPSRUnpacker::get_device_supported HAVE_CUDA" << endl;
  return dynamic_cast< CUDA::DeviceMemory*> ( memory );
#else
  return false;
#endif
}

//! Set the device on which the unpacker will operate
void dsp::CASPSRUnpacker::set_device (Memory* memory)
{
#if HAVE_CUDA
  CUDA::DeviceMemory* gpu = dynamic_cast< CUDA::DeviceMemory*>( memory );
  if (gpu)
  {
    staging.set_memory( memory );
    gpu_stream = (void *) gpu->get_stream();
  }

#else
  throw Error (InvalidState, "dsp::CASPSRUnpacker::set_device",
               "unsupported device");
#endif
}


bool dsp::CASPSRUnpacker::matches (const Observation* observation)
{
  return observation->get_machine()== "CASPSR"
    && observation->get_nbit() == 8;
}

void dsp::CASPSRUnpacker::unpack (uint64_t ndat,
                                  const unsigned char* from,
                                  const unsigned nskip,
                                  float* into,
                                  const unsigned fskip,
                                  unsigned long* hist)
{
  const float* lookup = table->get_values ();

#ifdef FAST_UNPACK

  const unsigned into_stride = fskip * 4;
  const unsigned from_stride = 8;

  for (uint64_t idat=0; idat < ndat; idat+=4)
  {
    into[0] = lookup[ from[0] ];
    into[1] = lookup[ from[1] ];
    into[2] = lookup[ from[2] ];
    into[3] = lookup[ from[3] ];

    from += from_stride;
    into += into_stride;
  }
#else
  int counter_four = 0;

  if (verbose)
    cerr << "dsp::CASPSRUnpacker::unpack ndat=" << ndat << endl;

  for (uint64_t idat = 0; idat < ndat; idat++)
  {
    hist[ *from ] ++;
    *into = lookup[ *from ];
    n_unpacked ++;
    
#ifdef _DEBUG
      cerr << idat << " " << int(*from) << " -> " << *into << endl;
#endif
    counter_four++;
    if (counter_four == 4)
      {
        counter_four = 0;
        from += 5; //(nskip+4);
      }
    else
      {
        from ++; //=nskip;
      }
    into += fskip;
  }
#endif

}

void dsp::CASPSRUnpacker::unpack ()
{
#if HAVE_CUDA
  if (gpu_stream != undefined_stream)
  {
    unpack_on_gpu ();
    return;
  }
#endif

  const uint64_t   ndat  = input->get_ndat();
  
  const unsigned nchan = input->get_nchan();
  const unsigned npol  = input->get_npol();
  const unsigned ndim  = input->get_ndim();
  
  const unsigned nskip = npol * nchan * ndim;
  const unsigned fskip = ndim;
  
  unsigned offset = 0;
  
  for (unsigned ichan=0; ichan<nchan; ichan++)
  {
    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      if (ipol==1)
        offset = 4;
      for (unsigned idim=0; idim<ndim; idim++)
      {
        const unsigned char* from = input->get_rawptr() + offset;
        float* into = output->get_datptr (ichan, ipol) + idim;
        unsigned long* hist = get_histogram (ipol);
              
#ifdef _DEBUG
        cerr << "c=" << ichan << " p=" << ipol << " d=" << idim << endl;
#endif
              
        unpack (ndat, from, nskip, into, fskip, hist);
        offset ++;
      }
    }
  }
}

unsigned dsp::CASPSRUnpacker::get_resolution () const { return 1024; }

#if HAVE_CUDA

void dsp::CASPSRUnpacker::unpack_on_gpu ()
{
  const uint64_t ndat = input->get_ndat();

  staging.Observation::operator=( *input );
  staging.resize(ndat);

  // staging buffer on the GPU for packed data
  unsigned char* d_staging = staging.get_rawptr();
 
  const unsigned char* from= input->get_rawptr();

  float* into_pola = output->get_datptr(0,0);
  float* into_polb = output->get_datptr(0,1);

  cudaStream_t stream = (cudaStream_t) gpu_stream;

  cudaError error;

  if (stream)
    error = cudaMemcpyAsync (d_staging, from, ndat*2,
                             cudaMemcpyHostToDevice, stream);
  else
    error = cudaMemcpy (d_staging, from, ndat*2, cudaMemcpyHostToDevice);

  if (error != cudaSuccess)
    throw Error (FailedCall, "CASPSRUnpacker::unpack_on_gpu",
                 "cudaMemcpy%s %s", stream?"Async":"", 
                 cudaGetErrorString (error));

  caspsr_unpack (stream, ndat, table->get_scale(), 
                 d_staging, into_pola, into_polb); 
}

#endif


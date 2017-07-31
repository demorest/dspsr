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

#include "dsp/CASPSRSingleUnpacker.h"
#include "dsp/BitTable.h"

#include "Error.h"

#if HAVE_CUDA
#include "dsp/MemoryCUDA.h"
#include "dsp/CASPSRUnpackerCUDA.h"
#include <cuda_runtime.h>
#endif

#include <errno.h>

using namespace std;

static void* const undefined_stream = (void *) -1;

dsp::CASPSRSingleUnpacker::CASPSRSingleUnpacker (const char* _name) : HistUnpacker (_name)
{
  if (verbose)
    cerr << "dsp::CASPSRSingleUnpacker ctor" << endl;

  set_nstate (256);
  gpu_stream = undefined_stream;

  table = new BitTable (8, BitTable::TwosComplement);

#if HAVE_CUDA
  int device;
  struct cudaDeviceProp gpu;
  cudaGetDevice(&device);
  cudaGetDeviceProperties (&gpu, device);
  threadsPerBlock = gpu.maxThreadsPerBlock;
#endif

  device_prepared = false;
}

dsp::CASPSRSingleUnpacker::~CASPSRSingleUnpacker ()
{
}

dsp::CASPSRSingleUnpacker * dsp::CASPSRSingleUnpacker::clone () const
{
  return new CASPSRSingleUnpacker (*this);
}

//! Return true if the unpacker can operate on the specified device
bool dsp::CASPSRSingleUnpacker::get_device_supported (Memory* memory) const
{
#if HAVE_CUDA
  if (verbose)
    cerr << "dsp::CASPSRSingleUnpacker::get_device_supported HAVE_CUDA" << endl;
  return dynamic_cast< CUDA::DeviceMemory*> ( memory );
#else
  return false;
#endif
}

//! Set the device on which the unpacker will operate
void dsp::CASPSRSingleUnpacker::set_device (Memory* memory)
{
#if HAVE_CUDA
  CUDA::DeviceMemory * gpu_mem = dynamic_cast< CUDA::DeviceMemory*>( memory );
  if (gpu_mem)
  {
    gpu_stream = (void *) gpu_mem->get_stream();
    if (verbose)
      cerr << "dsp::CASPSRSingleUnpacker::set_device using gpu memory" << endl;
  }
  else
  {
    if (verbose)
      cerr << "dsp::CASPSRSingleUnpacker::set_device using cpu memory" << endl;
    gpu_stream = undefined_stream;
    Unpacker::set_device (memory);
  }
#else
  Unpacker::set_device (memory);
#endif
  device_prepared = true;
}


bool dsp::CASPSRSingleUnpacker::matches (const Observation* observation)
{
  return observation->get_machine()== "CASPSR"
    && observation->get_nbit() == 8;
}

// default CPU unpacker for CASPSR format
void dsp::CASPSRSingleUnpacker::unpack_default ()
{
  uint64_t ndat = input->get_ndat();
  const float* lookup = table->get_values ();

  const uint64_t * from64 = (uint64_t *) input->get_rawptr();

  unsigned long* hist_p0 = get_histogram (0);
  unsigned long* hist_p1 = get_histogram (1);

  float * into_p0 = output->get_datptr (0, 0);
  float * into_p1 = output->get_datptr (0, 1);

  uint64_t val64;
  unsigned char * val8 = (unsigned char *) &val64;
  char * val8h = (char *) &val64;

  // process 4 samples, from 2 pols per loop
  for (uint64_t idat=0; idat<ndat; idat+=4)
  {
    // read 8 values
    val64 = *from64;

    into_p0[0] = lookup[ val8[0] ];
    into_p0[1] = lookup[ val8[1] ];
    into_p0[2] = lookup[ val8[2] ];
    into_p0[3] = lookup[ val8[3] ];

    into_p1[0] = lookup[ val8[4] ];
    into_p1[1] = lookup[ val8[5] ];
    into_p1[2] = lookup[ val8[6] ];
    into_p1[3] = lookup[ val8[7] ];

    hist_p0[int(val8h[0])+128]++;
    hist_p0[int(val8h[1])+128]++;
    hist_p0[int(val8h[2])+128]++;
    hist_p0[int(val8h[3])+128]++;

    hist_p1[int(val8h[4])+128]++;
    hist_p1[int(val8h[5])+128]++;
    hist_p1[int(val8h[6])+128]++;
    hist_p1[int(val8h[7])+128]++;

    from64  += 1;
    into_p0 += 4;
    into_p1 += 4;
  }
}

void dsp::CASPSRSingleUnpacker::unpack (uint64_t ndat,
                                  const unsigned char* from,
                                  float* into,
                                  const unsigned fskip,
                                  unsigned long* hist)
{
  if (verbose)
    cerr << "dsp::CASPSRSingleUnpacker::unpack(...)" << endl;
  const float* lookup = table->get_values ();
  const float scale = table->get_scale();

  const unsigned into_stride = fskip * 4;
  const unsigned from_stride = 2;

  // read 4 samples at a time
  uint32_t * from32 = (uint32_t *) from;
  uint32_t val32;
  unsigned char * val8 = (unsigned char *) &val32;

  //std::cout << ndat << std::endl;
  for (uint64_t idat=0; idat < ndat; idat+=4)
  {
    // read 4 uint8_t (actually int8_t)
    val32 = *from32;

    into[0] = lookup[ val8[0] ];
    into[1] = lookup[ val8[1] ];
    into[2] = lookup[ val8[2] ];
    into[3] = lookup[ val8[3] ];

    hist[val8[0]]++;
    hist[val8[1]]++;
    hist[val8[2]]++;
    hist[val8[3]]++;

    from32 += from_stride;
    into += into_stride;
  }
}

void dsp::CASPSRSingleUnpacker::unpack ()
{

#if HAVE_CUDA
  if (gpu_stream != undefined_stream)
  {
    unpack_on_gpu ();
    return;
  }
#endif

  // some programs (digifil) do not call set_device
  if (! device_prepared)
    set_device ( Memory::get_manager ());

  const uint64_t ndat  = input->get_ndat();
  const unsigned nchan = input->get_nchan();
  const unsigned npol  = input->get_npol();
  const unsigned ndim  = input->get_ndim();

  if (ndim == 1 && npol == 2 && nchan == 1)
  {
    unpack_default();
    return;
  }

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

        unpack (ndat, from, into, fskip, hist);
        offset ++;
      }
    }
  }
}

unsigned dsp::CASPSRSingleUnpacker::get_resolution () const { return 1024; }

#if HAVE_CUDA

void dsp::CASPSRSingleUnpacker::unpack_on_gpu ()
{
  const uint64_t ndat = input->get_ndat();
  const unsigned nchan = input->get_nchan();
  const unsigned ndim = input->get_ndim();
  const unsigned npol = input->get_npol();

  const unsigned char* from = input->get_rawptr();
  float * into_pola, * into_polb;
  unsigned ichan;

  cudaStream_t stream = (cudaStream_t) gpu_stream;
  cudaError error;

  for (ichan=0; ichan<nchan; ichan++)
  {
    into_pola = output->get_datptr(ichan, 0);
    into_polb = output->get_datptr(ichan, 1);

    caspsr_unpack (stream, ndat*ndim, table->get_scale(), 
                   from, into_pola, into_polb,
                   threadsPerBlock);

    from += ndat*ndim*npol;
  }
}

#endif


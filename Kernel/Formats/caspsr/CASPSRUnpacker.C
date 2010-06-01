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

using namespace std;

dsp::CASPSRUnpacker::CASPSRUnpacker (const char* _name) : HistUnpacker (_name)
{
  if (verbose)
    cerr << "dsp::CASPSRUnpacker ctor" << endl;

  set_nstate (256);
  gpu_stream = 0;

  table = new BitTable (8, BitTable::TwosComplement);
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
    gpu_stream = gpu->get_stream();
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
  int counter_four = 0;

  if (verbose)
    cerr << "dsp::CASPSRUnpacker::unpack ndat=" << ndat << endl;

  for (uint64_t idat = 0; idat < ndat; idat++)
  {
    hist[ *from ] ++;
    *into = lookup[ *from ];
    
    #ifdef _DEBUG
    cerr << idat << " " << int(*from) << "=" << *into << endl;
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
}

void dsp::CASPSRUnpacker::unpack ()
{
#if HAVE_CUDA
  if (gpu_stream)
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
  
  //cerr << "dsp::CASPSRUnpacker::unpack()" << endl;

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
 
  const unsigned char*  from= input->get_rawptr();

  float* into_pola = output->get_datptr(0,0);
  float* into_polb = output->get_datptr(0,1);

  cudaStream_t* stream = reinterpret_cast<cudaStream_t*>( gpu_stream );

  cudaError error = cudaMemcpyAsync (d_staging, from, ndat*2,
				     cudaMemcpyHostToDevice, *stream);

  if (error != cudaSuccess)
    cerr << "CASPSRUnpacker::unpack() cudaMemcpy FAIL: " 
	 << cudaGetErrorString (error) << endl;

  caspsr_unpack (ndat,table->get_scale(), d_staging, into_pola, into_polb); 
}

#endif


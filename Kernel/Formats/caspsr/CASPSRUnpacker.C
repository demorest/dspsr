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

#include <errno.h>

using namespace std;

static void* const undefined_stream = (void *) -1;

dsp::CASPSRUnpacker::CASPSRUnpacker (const char* _name) : HistUnpacker (_name)
{
  if (verbose)
    cerr << "dsp::CASPSRUnpacker ctor" << endl;

  set_nstate (256);
  gpu_stream = undefined_stream;

  table = new BitTable (8, BitTable::TwosComplement);

  n_threads = 0;
  context = 0;
  state = Idle;
  thread_count = 0;

  device_prepared = false;
}

dsp::CASPSRUnpacker::~CASPSRUnpacker ()
{
#if HAVE_CUDA
  if (n_threads)
  {
    stop_threads ();
    join_threads ();
  }
#endif
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
  CUDA::DeviceMemory * gpu_mem = dynamic_cast< CUDA::DeviceMemory*>( memory );
  if (gpu_mem)
  {
    gpu_stream = (void *) gpu_mem->get_stream();
#ifdef USE_TEXTURE_MEMORY
    if (verbose)
      cerr << "dsp::CASPSRUnpacker::set_device: using texture memory" << endl;
    CUDA::TextureMemory * texture_mem = new CUDA::TextureMemory (gpu_mem->get_stream());
    texture_mem->set_format_signed(8, 0, 0, 0);
    texture_mem->set_symbol("caspsr_unpack_tex");
    staging.set_memory( texture_mem );
#else
    if (verbose)
      cerr << "dsp::CASPSRUnpacker::set_device: using gpu memory" << endl;
    staging.set_memory( memory );
#endif
  }
  else
  {
    if (verbose)
      cerr << "dsp::CASPSRUnpacker::set_device: using cpu memory" << endl;
    gpu_stream = undefined_stream;
    n_threads = 2;
    context = new ThreadContext;
    state = Idle;

    thread_count = 0;
    ids.resize(n_threads);
    states.resize(n_threads);
    for (unsigned i=0; i<n_threads; i++)
    {
      if (verbose)
        cerr << "dsp::CASPSRUnpacker::set_device: starting cpu_unpacker_thread " << i << endl;
      states[i] = Idle;
      errno = pthread_create (&(ids[i]), 0, cpu_unpacker_thread, this);
      if (errno != 0)
        throw Error (FailedSys, "dsp::CASPSRUnpacker", "pthread_create");
    }
  }
#else
  Unpacker::set_device (memory);
#endif
  device_prepared = true;
}


bool dsp::CASPSRUnpacker::matches (const Observation* observation)
{
  return observation->get_machine()== "CASPSR"
    && observation->get_nbit() == 8;
}

void dsp::CASPSRUnpacker::unpack (uint64_t ndat,
                                  const unsigned char* from,
                                  float* into,
                                  const unsigned fskip,
                                  unsigned long* hist)
{
  cerr << "dsp::CASPSRUnpacker::unpack(...)" << endl;
  const float* lookup = table->get_values ();
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

  // some programs (digifil) do not call set_device
  if ( ! device_prepared )
    set_device ( Memory::get_manager ());

  start_threads();
  wait_threads();
}

void* dsp::CASPSRUnpacker::cpu_unpacker_thread (void* ptr)
{
  reinterpret_cast<CASPSRUnpacker*>( ptr )->thread ();
  return 0;
}

void dsp::CASPSRUnpacker::thread ()
{
  context->lock();

  unsigned thread_num = thread_count;
  thread_count++;

  // each thread unpacks 4 samples
  const unsigned ipol = thread_num % 2;
  const unsigned npol = 2;

  const unsigned into_stride = 4 * (int) (n_threads / npol);  // unpacked jump per thread iter
  const unsigned into_offset = 4 * (int) (thread_num / npol); // unpacked thread start offset

  const unsigned from_stride = 4 * n_threads;                 // raw jump per thread iter
  const unsigned from_offset = 4 * thread_num;                // raw thread start offset

  const float* lookup = table->get_values ();
  float * into = 0;

  while (state != Quit)
  {
    // wait for Active state
    while (states[thread_num] == Idle)
    {
      context->wait ();
    }

    if (states[thread_num] == Quit)
    {
      context->unlock();
      return;
    }
    context->unlock();


    // unpack ndat worth of data
    const uint64_t ndat  = input->get_ndat();
    const unsigned char* from = input->get_rawptr() + from_offset;
    into = output->get_datptr (0, ipol) + into_offset;

    for (uint64_t idat=into_offset; idat < ndat; idat+=into_stride)
    {
      into[0] = lookup[ from[0] ];
      into[1] = lookup[ from[1] ];
      into[2] = lookup[ from[2] ];
      into[3] = lookup[ from[3] ];

      from += from_stride;
      into += into_stride;
    }

    context->lock();
    states[thread_num] = Idle;

#ifdef _DEBUG 
      cerr << "thread[" << thread_num << "] done" << endl;
#endif
    context->broadcast();
  }
  context->unlock();
}

void dsp::CASPSRUnpacker::start_threads ()
{ 
  ThreadContext::Lock lock (context);
  
  while (state != Idle)
    context->wait ();
    
  for (unsigned i=0; i<n_threads; i++)
    states[i] = Active;
  state = Active;
  
  context->broadcast();
} 

void dsp::CASPSRUnpacker::wait_threads()
{ 
  ThreadContext::Lock lock (context);
  
  while (state == Active)
  { 
    bool all_idle = true;
    for (unsigned i=0; i<n_threads; i++)
    {
      if (states[i] != Idle)
        all_idle = false;
    }
    
    if (all_idle)
    {
      state = Idle;
    }
    else
      context->wait ();
  }
}

void dsp::CASPSRUnpacker::stop_threads ()
{
  ThreadContext::Lock lock (context);

  while (state != Idle)
    context->wait ();

  for (unsigned i=0; i<n_threads; i++)
    states[i] = Quit;
  state = Quit;

  context->broadcast();
}

void dsp::CASPSRUnpacker::join_threads ()
{
  void * result = 0;
  for (unsigned i=0; i<n_threads; i++)
    pthread_join (ids[i], &result);
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
#ifdef USE_TEXTURE_MEMORY
  if (verbose)
    cerr << "dsp::CASPSRUnpacker::unpack_on_gpu: creating TextureMemory" << endl;

  CUDA::TextureMemory * gpu_mem = dynamic_cast< CUDA::TextureMemory*>( staging.get_memory() );
  if (ndat > 0)
    gpu_mem->activate ( d_staging );
#endif
 
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


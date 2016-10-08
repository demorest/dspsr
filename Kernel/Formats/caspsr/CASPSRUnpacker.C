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

#if HAVE_CUDA
  int device;
  struct cudaDeviceProp gpu;
  cudaGetDevice(&device);
  cudaGetDeviceProperties (&gpu, device);
  threadsPerBlock = gpu.maxThreadsPerBlock;
#endif

  device_prepared = false;
  single_thread = true;
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
    if (verbose)
      cerr << "dsp::CASPSRUnpacker::set_device using gpu memory" << endl;
  }
  else
  {
    if (verbose)
      cerr << "dsp::CASPSRUnpacker::set_device using cpu memory" << endl;
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
  n_threads = 0;
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
  if (verbose)
    cerr << "dsp::CASPSRUnpacker::unpack(...)" << endl;
  const float* lookup = table->get_values ();
  const unsigned into_stride = fskip * 4;
  const unsigned from_stride = 8;

  //std::cout << ndat << std::endl;
  for (uint64_t idat=0; idat < ndat; idat+=4)
  {
    into[0] = lookup[ from[0] ]; //hist[from[0]]++;
    into[1] = lookup[ from[1] ]; //hist[from[1]]++;
    into[2] = lookup[ from[2] ]; //hist[from[2]]++;
    into[3] = lookup[ from[3] ]; //hist[from[3]]++;

    from += from_stride;
    into += into_stride;
  }
}

void dsp::CASPSRUnpacker::unpack_single_thread() {
  if (verbose)
    cerr << "dsp::CASPSRUnpacker::unpack(...)" << endl;
  const uint64_t ndat  = input->get_ndat();
  
  const unsigned nchan = input->get_nchan();
  const unsigned npol  = input->get_npol();
  const unsigned ndim  = input->get_ndim();
  
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

  if (n_threads) {
    start_threads();
    wait_threads();
  }
  else {
    unpack_single_thread();
  }
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


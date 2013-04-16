//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2013 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/MOPSRUnpacker.h"
#include "dsp/BitTable.h"

#include "Error.h"

#if HAVE_CUDA
#include "dsp/MemoryCUDA.h"
#include "dsp/MOPSRUnpackerCUDA.h"
#include <cuda_runtime.h>
#endif

#include <errno.h>

using namespace std;

static void* const undefined_stream = (void *) -1;

#ifdef _DEBUG
#define CHECK_ERROR(x) check_error(x)
#define CHECK_ERROR_STREAM(x,y) check_error_stream(x,y)
#else
#define CHECK_ERROR(x)
#define CHECK_ERROR_STREAM(x,y)
#endif

#if HAVE_CUDA
void check_error (const char *);
void check_error_stream (const char *, cudaStream_t);
#endif

dsp::MOPSRUnpacker::MOPSRUnpacker (const char* _name) : HistUnpacker (_name)
{
  if (verbose)
    cerr << "dsp::MOPSRUnpacker ctor" << endl;

  set_nstate (256);
  gpu_stream = undefined_stream;

  table = new BitTable (8, BitTable::TwosComplement);

#ifdef USE_UNPACK_THREADS
  n_threads = 0;
  context = 0;
  state = Idle;
  thread_count = 0;
#endif

  device_prepared = false;
}

dsp::MOPSRUnpacker::~MOPSRUnpacker ()
{
#if HAVE_CUDA
#ifdef USE_UNPACK_THREADS
  if (n_threads)
  {
    stop_threads ();
    join_threads ();
  }
#endif
#endif
}

//! Return true if the unpacker support the specified output order
bool dsp::MOPSRUnpacker::get_order_supported (TimeSeries::Order order) const
{
  return true;
}

//! Set the order of the dimensions in the output TimeSeries
void dsp::MOPSRUnpacker::set_output_order (TimeSeries::Order order)
{
  output_order = order;
}

dsp::MOPSRUnpacker * dsp::MOPSRUnpacker::clone () const
{
  return new MOPSRUnpacker (*this);
}

//! Return true if the unpacker can operate on the specified device
bool dsp::MOPSRUnpacker::get_device_supported (Memory* memory) const
{
#if HAVE_CUDA
  if (verbose)
    cerr << "dsp::MOPSRUnpacker::get_device_supported HAVE_CUDA" << endl;
  return dynamic_cast< CUDA::DeviceMemory*> ( memory );
#else
  return false;
#endif
}

//! Set the device on which the unpacker will operate
void dsp::MOPSRUnpacker::set_device (Memory* memory)
{
  cerr << "dsp::MOPSRUnpacker::set_device memory=" << memory << endl;
#if HAVE_CUDA
  CUDA::DeviceMemory * gpu_mem = dynamic_cast< CUDA::DeviceMemory*>( memory );
  if (gpu_mem)
  {
    gpu_stream = (void *) gpu_mem->get_stream();
    if (verbose)
      cerr << "dsp::MOPSRUnpacker::set_device gpu_stream=" << gpu_stream << endl;
#ifdef USE_TEXTURE_MEMORY
    CUDA::TextureMemory * texture_mem = new CUDA::TextureMemory (gpu_mem->get_stream());
    if (verbose)
      cerr << "dsp::MOPSRUnpacker::set_device using texture memory ptr=" << texture_mem << endl;
    texture_mem->set_format_signed(8, 8, 0, 0);

    cerr << "dsp::MOPSRUnpacker::set_device staging.set_memory (texture_mem)" << endl;
    staging.set_memory( texture_mem );
#else
    if (verbose)
      cerr << "dsp::MOPSRUnpacker::set_device: using gpu memory" << endl;
    staging.set_memory ( gpu_mem );
#endif
  }
  else
  {
    if (verbose)
      cerr << "dsp::MOPSRUnpacker::set_device: using cpu memory" << endl;
    gpu_stream = undefined_stream;
#ifdef USE_UNPACK_THREADS
    n_threads = 2;
    context = new ThreadContext;
    state = Idle;

    thread_count = 0;
    ids.resize(n_threads);
    states.resize(n_threads);
    for (unsigned i=0; i<n_threads; i++)
    {
      if (verbose)
        cerr << "dsp::MOPSRUnpacker::set_device: starting cpu_unpacker_thread " << i << endl;
      states[i] = Idle;
      errno = pthread_create (&(ids[i]), 0, cpu_unpacker_thread, this);
      if (errno != 0)
        throw Error (FailedSys, "dsp::MOPSRUnpacker", "pthread_create");
    }
#endif
  }
#else
  Unpacker::set_device (memory);
#endif
  device_prepared = true;
}


bool dsp::MOPSRUnpacker::matches (const Observation* observation)
{
  return observation->get_machine()== "MOPSR"
    && observation->get_state() == Signal::Analytic
    && observation->get_nbit() == 8 
    && observation->get_ndim() == 2
    && observation->get_npol() == 1;
}

/*
void dsp::MOPSRUnpacker::unpack (uint64_t ndat,
                                  const unsigned char* from,
                                  float* into,
                                  const unsigned fskip)
{
  cerr << "dsp::MOPSRUnpacker::unpack(...)" << endl;
  cerr << "dsp::MOPSRUnpacker::unpack ndat=" << ndat << " fskip=" << fskip << endl;;

  // * 2 for complex input/output
  const unsigned into_stride = fskip * 2;
  const unsigned from_stride = 2;
  const float* lookup = table->get_values ();

  for (uint64_t idat=0; idat < ndat; idat++)
  {
    into[0] = lookup[ from[0] ];
    into[1] = lookup[ from[1] ];

    from += from_stride;
    into += into_stride;
  }
}
*/

void dsp::MOPSRUnpacker::unpack ()
{
#if HAVE_CUDA
  if (gpu_stream != undefined_stream)
  {
    unpack_on_gpu ();
    return;
  }
#endif

#ifdef USE_UNPACK_THREADS
  // some programs (digifil) do not call set_device
  if ( ! device_prepared )
    set_device ( Memory::get_manager ());

  start_threads();
  wait_threads();
#else

  const unsigned int nchan = input->get_nchan();
  const unsigned int nant = input->get_nant();
  const unsigned int ndim = input->get_ndim();
  const unsigned int ipol = 0;
  const unsigned int npol = 1;
  const uint64_t ndat = input->get_ndat();

  // input channel stride - distance between successive (temporal) samples from same channel
  unsigned int in_chan_stride = nchan * nant * ndim;
  unsigned int out_chan_stride = ndim;
  //const float* lookup = table->get_values ();
  //const float scale = table->get_scale();

  if (verbose)
    cerr << "dsp::MOPSRUnpacker::unpack in_chan_stride="<< in_chan_stride << endl;

  switch ( output->get_order() )
  {
    case TimeSeries::OrderFPT:
    {
      for (unsigned ichan=0; ichan<nchan; ichan++)
      {
        const unsigned int in_chan_off =  ndim * nant * ichan;
        const int8_t * from = (int8_t *) (input->get_rawptr() + in_chan_off);
        float* into = output->get_datptr (ichan, ipol);
        for (unsigned idat=0; idat < ndat; idat++)
        {
          into[0] = float ( from[0] ); // Re
          into[1] = float ( from[1] ); // Im

          from += in_chan_stride;
          into += out_chan_stride;
        } 
      }
      break;
    }

    case TimeSeries::OrderTFP:
    {
      const int8_t * from = (int8_t *) input->get_rawptr();
      float* into = output->get_dattfp();
      const uint64_t nfloat = npol * nchan * ndat;

      for (uint64_t ifloat=0; ifloat < nfloat; ifloat++)
      {
        into[0] = (float) from[0];
        into[1] = (float) from[1];
        into += 2;
        from += 2;
      }
    }
    break;


    default:
      throw Error (InvalidState, "dsp::MOPSRUnpacker::unpack", "unrecognized order");

    break;
  }

#endif
}

#ifdef USE_UNPACK_THREADS
void* dsp::MOPSRUnpacker::cpu_unpacker_thread (void* ptr)
{
  reinterpret_cast<MOPSRUnpacker*>( ptr )->thread ();
  return 0;
}

void dsp::MOPSRUnpacker::thread ()
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

void dsp::MOPSRUnpacker::start_threads ()
{ 
  ThreadContext::Lock lock (context);
  
  while (state != Idle)
    context->wait ();
    
  for (unsigned i=0; i<n_threads; i++)
    states[i] = Active;
  state = Active;
  
  context->broadcast();
} 

void dsp::MOPSRUnpacker::wait_threads()
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

void dsp::MOPSRUnpacker::stop_threads ()
{
  ThreadContext::Lock lock (context);

  while (state != Idle)
    context->wait ();

  for (unsigned i=0; i<n_threads; i++)
    states[i] = Quit;
  state = Quit;

  context->broadcast();
}

void dsp::MOPSRUnpacker::join_threads ()
{
  void * result = 0;
  for (unsigned i=0; i<n_threads; i++)
    pthread_join (ids[i], &result);
}

#endif

unsigned dsp::MOPSRUnpacker::get_resolution () const { return 1024; }

#if HAVE_CUDA

void dsp::MOPSRUnpacker::unpack_on_gpu ()
{
  const uint64_t ndat = input->get_ndat();
  const uint64_t ndim = input->get_ndim();
  const uint64_t npol = input->get_npol();

  const uint64_t to_copy = ndat * ndim * npol;

  staging.Observation::operator=( *input );
  staging.resize(ndat);

  // staging buffer on the GPU for packed data
  unsigned char* d_staging = staging.get_rawptr();

  const unsigned char* from= input->get_rawptr();

  float* into = output->get_datptr(0,0);

  cudaStream_t stream = (cudaStream_t) gpu_stream;
  if (verbose)
    cerr << "dsp::MOPSRUnpacker::unpack_on_gpu stream=" << stream << endl;

  cudaError error;
  if (stream)
  {
    error = cudaMemcpyAsync (d_staging, from, to_copy, cudaMemcpyHostToDevice, stream);
    CHECK_ERROR_STREAM ("dsp::MOPSRUnpacker::unpack_on_gpu cudaMemcpyAsync", stream);
  }
  else
  {
    error = cudaMemcpy (d_staging, from, to_copy, cudaMemcpyHostToDevice);
    CHECK_ERROR ("dsp::MOPSRUnpacker::unpack_on_gpu cudaMemcpy");
  }
  

#ifdef USE_TEXTURE_MEMORY
  if (verbose)
    cerr << "dsp::MOPSRUnpacker::unpack_on_gpu binding TextureMemory" << endl;
  CUDA::TextureMemory * gpu_mem = dynamic_cast< CUDA::TextureMemory*>( staging.get_memory() );
  cerr << "dsp::MOPSRUnpacker::unpack_on_gpu textureMemory stream=" << stream << " gpu_mem->get_tex()= " << gpu_mem->get_tex() << endl;
#endif

  if (error != cudaSuccess)
    throw Error (FailedCall, "MOPSRUnpacker::unpack_on_gpu",
                 "cudaMemcpy%s %s", stream?"Async":"", 
                 cudaGetErrorString (error));

  if (verbose)
    cerr << "dsp::MOPSRUnpacker::unpack_on_gpu mopsr_unpack ndat=" << ndat << endl;
#ifdef USE_TEXTURE_MEMORY
  mopsr_unpack (stream, ndat, d_staging, into, gpu_mem->get_tex());
#else
  mopsr_unpack (stream, ndat, table->get_scale(), d_staging, into);
#endif

  if (stream)
    CHECK_ERROR_STREAM ("dsp::MOPSRUnpacker::unpack_on_gpu mopsr_unpack", stream);
  else
    CHECK_ERROR ("dsp::MOPSRUnpacker::unpack_on_gpu mopsr_unpack");
}

#endif

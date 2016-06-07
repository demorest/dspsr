//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2016 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/TimeSeriesCUDA.h"
#include "dsp/MemoryCUDA.h"

#include "Error.h"

void check_error_stream (const char*, cudaStream_t);

using namespace std;

void check_error (const char*);

template<typename T>
__global__ void copy_data_fpt_kernel(T * out, T* in,
                                     uint64_t ichanpol_stride,
                                     uint64_t ochanpol_stride,
                                     uint64_t ndat)
{
  uint64_t dx = blockIdx.x * blockDim.x + threadIdx.x;
  if (dx >= ndat)
    return;
  out[blockIdx.y * ochanpol_stride + dx] = in[blockIdx.y * ichanpol_stride + dx];
}

CUDA::TimeSeriesEngine::TimeSeriesEngine (dsp::Memory * _memory)
{
  memory = dynamic_cast<CUDA::DeviceMemory*>(_memory);
  buffer = NULL;
  buffer_size = 0;
}

CUDA::TimeSeriesEngine::~TimeSeriesEngine ()
{
  if (buffer)
    memory->do_free (buffer);
  buffer = 0;
}

void CUDA::TimeSeriesEngine::prepare (dsp::TimeSeries * parent)
{
  to = parent;
}

void CUDA::TimeSeriesEngine::prepare_buffer (unsigned nbytes)
{
  if (nbytes > buffer_size)
  {
    if (buffer)
      memory->do_free (buffer);
    buffer_size = nbytes;
    buffer = memory->do_allocate (buffer_size);
  }
}

// copy data from another time series to this time series
void CUDA::TimeSeriesEngine::copy_data_fpt (const dsp::TimeSeries* from, 
    uint64_t idat_start, uint64_t ndat)
{
  // cuda device that is executing this function
  int device;
  cudaGetDevice(&device);
  
#ifdef _DEBUG
  cerr << "CUDA::TimeSeriesEngine::copy_data_fpt from=" << (void *) from 
       << " idat_start=" << idat_start << " ndat=" << ndat << " device=" << device << endl;
#endif

  // stream and device upon which to TSE exists
  cudaStream_t to_stream = memory->get_stream();
  int to_device          = memory->get_device();

  // stream and device upon which from TSE exists
  const CUDA::DeviceMemory * from_mem = dynamic_cast<const CUDA::DeviceMemory*>( from->get_memory());
  cudaStream_t from_stream = from_mem->get_stream();
  const int from_device    = from_mem->get_device();

  if (!from_mem)
    throw Error (FailedSys, "CUDA::TimeSeriesEngine::copy_data_fpt", "From TimeSeries did not use DeviceMemory");

  unsigned nchan = to->get_nchan();
  unsigned npol  = to->get_npol();
  unsigned ndim  = to->get_ndim();
  
  uint64_t ichanpol_stride = 0;
  uint64_t ochanpol_stride = 0;
  uint64_t bchanpol_stride = ndat;

  if (npol > 1)
  {
    ochanpol_stride = to->get_datptr (0,1) - to->get_datptr (0,0);
    ichanpol_stride = from->get_datptr (0,1) - from->get_datptr (0,0);
  }
  else if (nchan > 1)
  {
    ochanpol_stride = to->get_datptr (1,0) - to->get_datptr (1,0);
    ichanpol_stride = from->get_datptr (1,0) - from->get_datptr (1,0);
  }
  else
  {
    ; 
  }

  ichanpol_stride /= ndim;
  ochanpol_stride /= ndim;

#ifdef _DEBUG
  cerr << "CUDA::TimeSeriesEngine::copy_data_fpt streams to="
       << (void*) to_stream << " from=" << (void*) from_stream << endl;
  cerr << "CUDA::TimeSeriesEngine::copy_data_fpt device to=" << device 
       << " from=" << from_device << endl;
#endif

  unsigned nthread = 1024;
  if (nthread > ndat)
    nthread = ndat;
  dim3 blocks = dim3 (ndat / nthread, nchan*npol);
  if (ndat % nthread)
    blocks.x++;

  size_t nbytes = nchan * ndim * npol * ndat * sizeof(float);

  // to un-stride from to the buffer
  if (to_stream != from_stream)
  {
    // now ensure the to TSE is of sufficient size
    if (device != to_device)
      cudaSetDevice (to_device);
    prepare_buffer (nbytes);

    // switch to the from_device to ensure buffer is allocated
    if (to_device != from_device)
      cudaSetDevice (from_device);

    CUDA::TimeSeriesEngine * from_engine = dynamic_cast<CUDA::TimeSeriesEngine*>(from->get_engine());

    from_engine->prepare_buffer (nbytes);

    // copy from -> buffer
    if (ndim == 2)
    {
      float2 * to_ptr   = (float2 *) from_engine->buffer;
      float2 * from_ptr = (float2 *) from->get_datptr (0,0);
      copy_data_fpt_kernel<float2><<<blocks,nthread,0,from_stream>>> (
        to_ptr, from_ptr + idat_start, ichanpol_stride, bchanpol_stride, ndat);
    }
    else
    {
      float * to_ptr   = (float *) from_engine->buffer;
      float * from_ptr = (float *) from->get_datptr (0,0);
      copy_data_fpt_kernel<float><<<blocks,nthread,0,from_stream>>> (
        to_ptr, from_ptr + idat_start, ichanpol_stride, bchanpol_stride, ndat);
    }

    if (to_device != from_device)
    {
      cudaMemcpyPeer (buffer, device, from_engine->buffer, to_device, nbytes);
    }
    else
    {
      cudaMemcpy(buffer, from_engine->buffer, nbytes, cudaMemcpyDeviceToDevice);
    }

    // switch to the from_device to ensure buffer is allocated
    if (to_device != from_device)
      cudaSetDevice (to_device);

    // copy buffer -> to
    if (ndim == 2)
    {
      float2 * to_ptr   = (float2 *) to->get_datptr (0,0);
      float2 * from_ptr = (float2 *) buffer;
      copy_data_fpt_kernel<float2><<<blocks,nthread,0,to_stream>>> (
          to_ptr, from_ptr, bchanpol_stride, ochanpol_stride, ndat);
    }
    else
    {
      float * to_ptr   = (float *) to->get_datptr (0,0);
      float * from_ptr = (float *) buffer;
      copy_data_fpt_kernel<float><<<blocks,nthread,0,to_stream>>> (
        to_ptr, from_ptr, bchanpol_stride, ochanpol_stride, ndat);
    }
  }
  // in the same stream & device
  else
  {
    if (ndim == 2)
    {
      float2 * to_ptr   = (float2 *) to->get_datptr (0,0);
      float2 * from_ptr = (float2 *) from->get_datptr (0,0);
      copy_data_fpt_kernel<float2><<<blocks,nthread,0,to_stream>>> (
        to_ptr, from_ptr + idat_start, ichanpol_stride, ochanpol_stride, ndat);
    }
    else
    {
      float * to_ptr   = (float *) to->get_datptr (0,0);
      float * from_ptr = (float *) from->get_datptr (0,0);
      copy_data_fpt_kernel<float><<<blocks,nthread,0,to_stream>>> (
        to_ptr, from_ptr + idat_start, ichanpol_stride, ochanpol_stride, ndat);
    }
  }

  cudaSetDevice(device);
}


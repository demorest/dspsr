//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2016 by Andre Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SKFilterbankCUDA.h"
#include "dsp/MemoryCUDA.h"

#include "CUFFTError.h"
#include "Error.h"
#include "templates.h"
#include "debug.h"

#include <stdio.h>
#include <memory>
#include <string.h>

//#define _DEBUG 1

using namespace std;

void check_error_stream (const char*, cudaStream_t);

/* Perform a reduction including SQLD calculations */
__global__ void reduce_sqld (cufftComplex* input, cufftComplex* output, float * skout, unsigned nchan, unsigned npol, unsigned M)
{
  // each block is a tsrunch, threads are channels

  // increment input and output pointer
  input  += (blockIdx.x * nchan * M);
  output += (blockIdx.x * nchan);
  skout  += (blockIdx.x * nchan);

  const float M_fac = (M+1) / (M-1);

  cufftComplex val;
  for (unsigned ichan=threadIdx.x; ichan<nchan; ichan+=blockDim.x)
  {
    float s1 = 0;
    float s2 = 0;
    cufftComplex* in = input;
    for (unsigned idat=0; idat<M; idat++)
    {
      val = in[ichan];
      float power = ((val.x * val.x) + (val.y * val.y));
      s1 += power;
      s2 += power * power;
      in += nchan;
    }
    val.x = s1;
    val.y = s2;
    output[ichan] = val;

    // write out the SK estimate for block of M
    skout[npol*ichan] = M_fac * (M * (s2 / (s1 * s1)) - 1);
  }
}

/* sum each set of S1 and S2 and compute SK estimate for whole block */
__global__ void reduce_sk_estimate (cufftComplex* input, float * output, unsigned nchan, unsigned npol, unsigned ndat, float M)
{
  // input are stored in TF order
  cufftComplex val;
  const float M_fac = (M+1) / (M-1);

  for (unsigned ichan=threadIdx.x; ichan<nchan; ichan+=blockDim.x)
  {
    float s1 = 0;
    float s2 = 0;
    cufftComplex* in = input;

    for (unsigned idat=0; idat<ndat; idat++)
    {
      val = in[ichan];
      s1 += val.x;
      s2 += val.y;
      in += nchan;
    }
    output[npol*ichan] = M_fac * (M * (s2 / (s1 * s1)) - 1);
  }
}

CUDA::SKFilterbankEngine::SKFilterbankEngine (dsp::Memory * _memory, unsigned _tscrunch)
{
  memory = dynamic_cast<CUDA::DeviceMemory*>(_memory);
  stream = memory->get_stream();
  tscrunch = _tscrunch;

  cufftResult result = cufftCreate (&plan);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::SKFilterbankEngine::SKFilterbankEngine",
                      "cufftCreate(plan)");
  npt = 0;
}

CUDA::SKFilterbankEngine::~SKFilterbankEngine ()
{
}

void CUDA::SKFilterbankEngine::setup ()
{
  if (dsp::Operation::verbose)
    cerr << "CUDA::SKFilterbankEngine::setup ()" << endl;

  // determine GPU capabilities
  int device = 0;
  cudaGetDevice(&device);
  struct cudaDeviceProp device_properties;
  cudaGetDeviceProperties (&device_properties, device);
  max_threads_per_block = device_properties.maxThreadsPerBlock;
}

void CUDA::SKFilterbankEngine::prepare (const dsp::TimeSeries * input, unsigned _npt)
{
  // real or complex input
  cufftType type = CUFFT_C2C;
  if (input->get_state() == Signal::Nyquist)
    type = CUFFT_R2C;

  npt = _npt;

  unsigned ndim = input->get_ndim();
  uint64_t ndat = input->get_ndat();
  unsigned nbatch = (ndat / npt);

  // 1D transform
  int rank = 1;
  int inembed[1] = { npt };
  int onembed[1] = { npt / ndim };

  // distance between successive elements
  int istride = 1;
  int ostride = 1;

  // distance between sucessive batches
  int idist = npt;
  int odist = npt / ndim;
  nchan = odist;

  size_t work_size;

  cufftResult result = cufftMakePlanMany (plan, rank, &npt,
                              inembed, istride, idist,
                              onembed, ostride, odist,
                              type, nbatch, &work_size);

  result = cufftSetStream (plan, stream);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::SKFilterbankEngine::prepare",
          "cufftSetStream(plan)");

  size_t bytes_required = nbatch * nchan * sizeof (cufftComplex);
  if (bytes_required > buffer_size)
  {
    if (buffer) 
      memory->do_free (buffer);
    buffer = memory->do_allocate (bytes_required);
    buffer_size = bytes_required;
  }

  bytes_required = (nbatch / tscrunch) * nchan * sizeof(cufftComplex);
  if (bytes_required > sums_size)
  {
    if (sums)
      memory->do_free (sums);
    sums = memory->do_allocate (sums_size);
    sums_size = bytes_required;
  }
}

void CUDA::SKFilterbankEngine::perform (const dsp::TimeSeries* input,
                                        dsp::TimeSeries* output,
                                        dsp::TimeSeries* output_tscr)
{
  if (dsp::Operation::verbose)
    std::cerr << "CUDA::SKFilterbankEngine::perform()" << std::endl;

  uint64_t ndat  = input->get_ndat();
  unsigned npol  = input->get_npol ();
  unsigned npart = (unsigned) (ndat / npt);

  if (input->get_order() != dsp::TimeSeries::OrderFPT)
    throw Error(InvalidState, "CUDA::SKFilterbankEngine::perform",
                "Only OrderFPT input order is supported");

  // TODO decide what to do about multi-input channel data

  // adjust FFT plan if required, TODO work on how npt is passed
  if (npart != nbatch)
    prepare (input, npt);
   
  // FFT output buffer from batched FFT
  cufftComplex * buf = (cufftComplex *) buffer;

  unsigned input_nchan = input->get_nchan ();
  if (dsp::Operation::verbose)
    std::cerr << "CUDA::SKFilterbankEngine::perform ndat=" << ndat 
              << " input_nchan=" << input_nchan << " output_nchan=" << nchan 
              << " npol=" << npol << " tscrunch=" << tscrunch << std::endl;

  for (unsigned ipol=0; ipol<npol; ipol++)
  {
    // input time series
    float * in = (float *) input->get_datptr (0, ipol);

    // output SK estimates at (1/M) time sampling
    float * out = (float *) output->get_dattfp();

    // output SK estimates at block resolution
    float * out_tscr = (float *) output_tscr->get_dattfp();

    // batch FFT all the input data
    if (type == CUFFT_R2C)
      fft_real ((cufftReal *) in, buf);
    else
      fft_complex ((cufftComplex *)in, buf);

    // specta now exist in out in TF format
    int nthread = nchan;    
    int nblocks = nbatch;

    // convert the spectra into tscrunched S1 and S2 sums in Re and Im
    reduce_sqld<<<nblocks,nthread,0,stream>>> (buf, (cufftComplex *) sums, out + ipol, nchan, npol, tscrunch);

    // compute a tscrunched output SK
    reduce_sk_estimate<<<1,nthread,0,stream>>>((cufftComplex *) sums, out_tscr + ipol, nchan, npol, npart, tscrunch);
  }

  check_error_stream("CUDA::SKFilterBank::perform", stream);

}

void CUDA::SKFilterbankEngine::fft_real (cufftReal *in, cufftComplex * out)
{
  cufftResult result = cufftExecR2C (plan, in, out);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::SKFilterbankEngine::fft_real",
                      "cufftExecR2C(plan)");
}

void CUDA::SKFilterbankEngine::fft_complex (cufftComplex *in, cufftComplex * out)
{
  cufftResult result = cufftExecC2C (plan, in, out, CUFFT_FORWARD);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::SKFilterbankEngine::fft_complex",
                      "cufftExecC2C(plan)");
}


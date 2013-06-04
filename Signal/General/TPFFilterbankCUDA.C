//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2012 by Willem van Straten and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#define _DEBUG 1

#include "dsp/TPFFilterbankCUDA.h"
#include "debug.h"

#include <cuda_runtime.h>

#include <iostream>
using namespace std;

CUDA::TPFFilterbankEngine::TPFFilterbankEngine (cudaStream_t _stream)
{
  real_to_complex = false;

  nchan = 0;

  d_fft = 0;

  stream = _stream;

  plan_fwd = 0;

  plan_prepared = false;

  nbatch = 1;
}

CUDA::TPFFilterbankEngine::~TPFFilterbankEngine ()
{
}

void CUDA::TPFFilterbankEngine::setup (dsp::Filterbank* filterbank)
{

  cerr << "CUDA::TPFFilterbankEngine::setup" << endl;
  nchan = filterbank->get_nchan ();

  real_to_complex = (filterbank->get_input()->get_state() == Signal::Nyquist);

  DEBUG("CUDA::TPFFilterbankEngine::setup nchan=" << nchan);

  //unsigned data_size = nchan * 2;
  //DEBUG("CUDA::TPFFilterbankEngine::setup data_size=" << data_size);

  // determine GPU capabilities 
  int device = 0;
  cudaGetDevice(&device);
  struct cudaDeviceProp device_properties;
  cudaGetDeviceProperties (&device_properties, device);
  max_threads_per_block = device_properties.maxThreadsPerBlock;

  if (nbatch)
  {
    DEBUG("CUDA::TPFFilterbankEngine::setup delaying plan creation");
    plan_prepared = false;
  }
  else
  {
    DEBUG("CUDA::TPFFilterbankEngine::setup create_plan()");
    create_plan ();
    plan_prepared = true;
  }
}

void CUDA::TPFFilterbankEngine::create_plan ()
{
  if (real_to_complex)
  {
    DEBUG("CUDA::TPFFilterbankEngine::create_plan R2C plan size=" << nchan * 2);
    cufftPlan1d (&plan_fwd, nchan * 2, CUFFT_R2C, 1);
  }
  else
  {
    DEBUG("CUDA::TPFFilterbankEngine::create_plan C2C plan size=" << nchan);
    cufftPlan1d (&plan_fwd, nchan, CUFFT_C2C, 1);
  }

  DEBUG("CUDA::TPFFilterbankEngine::create_plan setting stream=" << stream);
  cufftSetStream (plan_fwd, stream);

  // optimal performance for CUFFT regarding data layout
  cufftSetCompatibilityMode(plan_fwd, CUFFT_COMPATIBILITY_NATIVE);
}

void CUDA::TPFFilterbankEngine::create_batched_plan (uint64_t npart, uint64_t in_step, uint64_t out_step)
{
  cerr << "CUDA::TPFFilterbankEngine::create_batched_plan (" << npart << "," << in_step << "," << out_step  << ")" << endl;

  if (plan_prepared)
  {
    cerr << "CUDA::TPFFilterbankEngine::create_batched_plan plan already prepared!!" << endl;
    return;
  }

  const int rank = 1;
  int batch = npart;
  int npoints[1];
  cufftType type;

  if (real_to_complex)
  {
    type = CUFFT_R2C;
    npoints[0] = nchan * 2;
  }
  else
  {
    type = CUFFT_C2C; 
    npoints[0] = nchan;
  }

  int inembed[1];
  inembed[0] = npoints[0];
  int istride = 1;
  int idist = (int) in_step;

  int onembed[1];
  onembed[0] = nchan;
  int ostride = 1;
  int odist = (int) out_step;

  // create the plan
  cufftResult result = cufftPlanMany(&plan_fwd, rank, npoints, inembed, istride, idist, onembed, ostride, odist, type, batch);

  DEBUG("CUDA::TPFFilterbankEngine::setup setting stream=" << stream);
  cufftSetStream (plan_fwd, stream);

  // optimal performance for CUFFT regarding data layout
  cufftSetCompatibilityMode(plan_fwd, CUFFT_COMPATIBILITY_NATIVE);

  plan_prepared = true;

}

extern void check_error (const char*);

void CUDA::TPFFilterbankEngine::finish ()
{
  check_error ("CUDA::TPFFilterbankEngine::finish");
}

void CUDA::TPFFilterbankEngine::perform (const dsp::TimeSeries * in, 
                                         dsp::TimeSeries * out,
                                         uint64_t npart, 
                                         const uint64_t in_step,
                                         const uint64_t out_step)
{
  //dsp::TimeSeries* out, uint64_t npart, uint64_t in_step, uint64_t out_step)

  const unsigned npol = in->get_npol();
  const uint64_t input_ndat = in->get_ndat();
  const unsigned input_nchan = in->get_nchan();
  const unsigned output_nchan = out->get_nchan();

  if (!plan_prepared && nbatch > 0)
  {
    cerr << "CUDA::TPFFilterbankEngine::perform creating batched plan!" << endl;
    create_batched_plan (npart, in_step, out_step);
  }

  if (dsp::Operation::verbose)
  {
    DEBUG("CUDA::TPFFilterbankEngine::perform in_step=" << in_step);
    DEBUG("CUDA::TPFFilterbankEngine::perform out_step=" << out_step);
    DEBUG("CUDA::TPFFilterbankEngine::perform npart=" << npart);
    DEBUG("CUDA::TPFFilterbankEngine::perform npol=" << npol);
    DEBUG("CUDA::TPFFilterbankEngine::perform input ndat=" << input_ndat);
    DEBUG("CUDA::TPFFilterbankEngine::perform input nchan=" << input_nchan);
    DEBUG("CUDA::TPFFilterbankEngine::perform output nchan=" << output_nchan);
  }

  uint64_t in_offset, out_offset;

  float * input_ptr;
  float2 * input_ptr2;
  float2 * output_ptr;

  for (unsigned ichan=0; ichan<input_nchan; ichan++)
  {
    for (unsigned ipol=0; ipol < npol; ipol++)
    {
      for (unsigned ipart=0; ipart < npart; ipart++)
      {
        in_offset = ipart * in_step;
        out_offset = ipart * out_step;

        input_ptr = const_cast<float*>(in->get_datptr (ichan, ipol)) + in_offset;
        output_ptr = (float2 *) (out->get_dattpf () + out_offset);

        if (real_to_complex)
        {
          if (dsp::Operation::verbose)
            DEBUG("CUDA::TPFFilterbankEngine::perform cufftExecR2C");
          cufftExecR2C(plan_fwd, input_ptr, output_ptr);
          check_error ("CUDA::TPFFilterbankEngine::perform cufftExecR2C");
        }
        else
        {
          input_ptr2 = (float2*) input_ptr;
          if (dsp::Operation::verbose)
            DEBUG("CUDA::TPFFilterbankEngine::perform cufftExecC2C");
          cufftExecC2C(plan_fwd, input_ptr2, output_ptr, CUFFT_FORWARD);
          check_error ("CUDA::TPFFilterbankEngine::perform cufftExecC2C");
        }

        if (nbatch > 0)
          ipart = npart;
      }
    }
  }
       
  if (verbose)
    check_error ("CUDA::TPFFilterbankEngine::perform");

}

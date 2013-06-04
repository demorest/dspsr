//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2012 by Willem van Straten and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

//#define _DEBUG 1

#include "dsp/SpatialFilterbankCUDA.h"
#include "debug.h"

#include <cuda_runtime.h>

#include <iostream>

void check_error (const char*);
void check_error_stream (const char*, cudaStream_t);

#ifdef _DEBUG
#define CHECK_ERROR(x) check_error(x)
#define CHECK_ERROR_STREAM(x,y) check_error_stream(x,y)
#else
#define CHECK_ERROR(x)
#define CHECK_ERROR_STREAM(x,y)
#endif

using namespace std;

CUDA::SpatialFilterbankEngine::SpatialFilterbankEngine (cudaStream_t _stream)
{
  real_to_complex = false;

  nx = 0;
  ny = 0;

  stream = _stream;

  plan_fwd = 0;

  plan_prepared = false;

  nbatch = 1;
}

CUDA::SpatialFilterbankEngine::~SpatialFilterbankEngine ()
{
}

void CUDA::SpatialFilterbankEngine::setup (dsp::Filterbank* filterbank)
{
  cerr << "CUDA::SpatialFilterbankEngine::setup (" << filterbank << ")" << endl;

  ny = filterbank->get_nchan ();
  nx = filterbank->get_freq_res ();

  DEBUG("CUDA::SpatialFilterbankEngine::setup nx=" << nx << " ny=" << ny);

  real_to_complex = (filterbank->get_input()->get_state() == Signal::Nyquist);

  if (nbatch)
  {
    DEBUG("CUDA::SpatialFilterbankEngine::setup delaying plan creation");
  }
  else
  {
    DEBUG("CUDA::SpatialFilterbankEngine::setup create_plan()");
    create_plan ();
    plan_prepared = true;
  }
}

void CUDA::SpatialFilterbankEngine::set_scratch (float * _scratch)
{
  scratch = _scratch;
}

void CUDA::SpatialFilterbankEngine::create_plan ()
{
  cerr << "CUDA::SpatialFilterbankEngine::create_plan R2C plan size=[" << (nx * 2) << ", " << ny << "]" << endl;

  if (real_to_complex)
  {
    DEBUG("CUDA::SpatialFilterbankEngine::create_plan R2C plan size=[" << (nx * 2) << ", " << ny << "]");
    cufftPlan2d (&plan_fwd, nx * 2, ny, CUFFT_R2C);
  }
  else
  {
    DEBUG("CUDA::SpatialFilterbankEngine::create_plan C2C plan size=[" << nx << ", " << ny << "]");
    cufftPlan2d (&plan_fwd, nx, ny, CUFFT_C2C);
  }

  DEBUG("CUDA::SpatialFilterbankEngine::create_plan setting stream=" << stream);
  cufftSetStream (plan_fwd, stream);

  // optimal performance for CUFFT regarding data layout
  cufftSetCompatibilityMode(plan_fwd, CUFFT_COMPATIBILITY_NATIVE);
}

void CUDA::SpatialFilterbankEngine::create_batched_plan (uint64_t npart, unsigned npol, uint64_t in_step, uint64_t out_step)
{
  cerr <<"CUDA::SpatialFilterbankEngine::create_batched_plan (" << npart << ", " << npol << ", " << in_step << "," << out_step  << ")" << endl;

  if (plan_prepared)
  {
    cerr << "CUDA::SpatialFilterbankEngine::create_batched_plan plan already prepared!!" << endl;
    return;
  }

  const int rank = 2;
  int n[rank];
  int inembed[rank];
  int onembed[rank];

  int samples_per_dat = 1;
  cufftType type;

  if (real_to_complex)
  {
    type = CUFFT_R2C;
    n[0] = ny;
    n[1] = nx * 2;
  }
  else
  {
    type = CUFFT_C2C; 
    n[0] = ny;
    n[1] = nx;
  }

  inembed[1] = n[1];
  inembed[0] = n[0];
  onembed[1] = n[1];
  onembed[0] = n[0];
  //cerr << "CUDA::SpatialFilterbankEngine::create_batched_plan inembed=[" << inembed[0] << ", " << inembed[1] << "]" << endl;
  //cerr << "CUDA::SpatialFilterbankEngine::create_batched_plan onembed=[" << onembed[0] << ", " << onembed[1] << "]" << endl;

  int istride = 1;
  int ostride = 1;    // polarizations will be interleaved in output
  //cerr << "CUDA::SpatialFilterbankEngine::create_batched_plan stride in=" << istride << " out=" << ostride << endl;

  int idist = n[0] * n[1];
  int odist = n[0] * n[1];
  //cerr << "CUDA::SpatialFilterbankEngine::create_batched_plan dist in=" << idist << " out=" << odist << endl;

  int batch = npart;
  //cerr << "CUDA::SpatialFilterbankEngine::create_batched_plan n=[" << n[1] << "," << n[0] << "] batch=" << batch << endl;

  // create the plan
  //cufftResult result = cufftPlanMany(&plan_fwd, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch);
  cufftResult result = cufftPlanMany(&plan_fwd, rank, n, NULL, istride, idist, NULL, ostride, odist, type, batch);
  if (result != CUFFT_SUCCESS)
    cerr << "CUDA::SpatialFilterbankEngine::create_batched_plan CUFFT error: Plan creation failed!" << endl;

  DEBUG("CUDA::SpatialFilterbankEngine::setup setting stream=" << stream);
  result = cufftSetStream (plan_fwd, stream);
  if (result != CUFFT_SUCCESS)
    cerr << "CUDA::SpatialFilterbankEngine::create_batched_plan CUFFT error: Failed to set CUFFT stream" << endl;

  // optimal performance for CUFFT regarding data layout
  result = cufftSetCompatibilityMode(plan_fwd, CUFFT_COMPATIBILITY_NATIVE);
  if (result != CUFFT_SUCCESS)
    cerr << "CUDA::SpatialFilterbankEngine::create_batched_plan CUFFT error:  Failed to set compatibility mode" << endl;

  plan_prepared = true;

}

void CUDA::SpatialFilterbankEngine::finish ()
{
  DEBUG("CUDA::SpatialFilterbankEngine::finish calling cudaStreamSynchronize()");
  cudaStreamSynchronize (stream);
}

void CUDA::SpatialFilterbankEngine::perform (const dsp::TimeSeries* in, dsp::TimeSeries* out,
                  uint64_t npart, uint64_t in_step, uint64_t out_step)
{
  CHECK_ERROR_STREAM ("CUDA::SpatialFilterbankEngine::perform cufftExecC2C", stream);

  const unsigned npol = in->get_npol();
  const uint64_t input_ndat = in->get_ndat();
  const unsigned input_nchan = in->get_nchan();

  DEBUG("CUDA::SpatialFilterbankEngine::perform()");

  if (!plan_prepared && nbatch > 0)
  {
    //cerr << "CUDA::SpatialFilterbankEngine::perform creating batched npart=" << npart << endl;
    create_batched_plan (npart, npol, in_step, out_step);
  }

  if (dsp::Operation::verbose)
  {
    DEBUG("CUDA::SpatialFilterbankEngine::perform in_step=" << in_step);
    DEBUG("CUDA::SpatialFilterbankEngine::perform out_step=" << out_step);
    DEBUG("CUDA::SpatialFilterbankEngine::perform npart=" << npart);
    DEBUG("CUDA::SpatialFilterbankEngine::perform npol=" << npol);
    DEBUG("CUDA::SpatialFilterbankEngine::perform input ndat=" << input_ndat);
    DEBUG("CUDA::SpatialFilterbankEngine::perform input nchan=" << input_nchan);
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
        out_offset = (ipart * out_step) + (ipol * 2);

        //DEBUG("CUDA::SpatialFilterbankEngine::perform offsets ipol=" << ipol << " ichan=" << ichan << " input=" << in_offset << " output=" << out_offset);
    
        input_ptr = const_cast<float*>(in->get_datptr (ichan, ipol)) + in_offset;
        output_ptr = (float2 *) (out->get_dattfp () + out_offset);

        //DEBUG("CUDA::SpatialFilterbankEngine::perform input_ptr=" << (void*) input_ptr << ", output_ptr=" << (void *) output_ptr);

        if (real_to_complex)
        {
          //if (dsp::Operation::verbose)
            DEBUG("CUDA::SpatialFilterbankEngine::perform cufftExecR2C");
          cufftResult result = cufftExecR2C(plan_fwd, input_ptr, output_ptr);
          CHECK_ERROR_STREAM ("CUDA::SpatialFilterbankEngine::perform cufftExecR2C", stream);
          //CHECK_ERROR ("CUDA::SpatialFilterbankEngine::perform cufftExecR2C");
        }
        else
        {
          input_ptr2 = (float2*) input_ptr;
          if (dsp::Operation::verbose)
            DEBUG("CUDA::SpatialFilterbankEngine::perform cufftExecC2C");
          //CHECK_ERROR_STREAM ("CUDA::SpatialFilterbankEngine::perform pre cufftExecC2C", stream);
          cufftResult result = cufftExecC2C(plan_fwd, input_ptr2, output_ptr, CUFFT_FORWARD);
          //CHECK_ERROR_STREAM ("CUDA::SpatialFilterbankEngine::perform post cufftExecC2C", stream);
          if (result != CUFFT_SUCCESS)
            cerr << "CUDA::SpatialFilterbankEngine::perform cufftExecC2C failed: " << result << endl;
          CHECK_ERROR_STREAM ("CUDA::SpatialFilterbankEngine::perform cufftExecC2C", stream);
        }

        if (nbatch > 0)
          ipart = npart;
      }
    }
  }
  CHECK_ERROR_STREAM ("CUDA::SpatialFilterbankEngine::perform", stream);
  //CHECK_ERROR ("CUDA::SpatialFilterbankEngine::perform");
}

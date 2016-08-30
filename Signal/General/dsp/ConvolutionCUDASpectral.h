//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2015 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __baseband_cuda_ConvolutionSpectral_h
#define __baseband_cuda_ConvolutionSpectral_h

#include <cufft.h>
#include <config.h>

#include "dsp/Convolution.h"
#include "dsp/LaunchConfig.h"

namespace CUDA
{
  class ConvolutionEngineSpectral : public dsp::Convolution::Engine
  {
  public:

    //! Default Constructor
    ConvolutionEngineSpectral (cudaStream_t stream);
    ~ConvolutionEngineSpectral();

    void regenerate_plans();

    void set_scratch (void * scratch);

    //! prepare the required attributes for the engine
    void prepare (dsp::Convolution * convolution);

    //! setup the dedispersion kernel from the response
    void setup_kernel (const dsp::Response * response);

    //! configure batched FFT
    void setup_batched (const dsp::TimeSeries* input, dsp::TimeSeries * output);

#if HAVE_CUFFT_CALLBACKS
    //! setup FFT callbacks
    void setup_callbacks ();
#endif

    void perform (const dsp::TimeSeries* input, dsp::TimeSeries* output,
                  unsigned npart);

  protected:

    void perform_complex (const dsp::TimeSeries* input, dsp::TimeSeries * output,
                         unsigned npart);

    void perform_real (const dsp::TimeSeries* input, dsp::TimeSeries * output,
                       unsigned npart);

    cudaStream_t stream;

    LaunchConfig1D mp;

    cufftType type_fwd;

    cufftHandle plan_fwd;

    cufftHandle plan_bwd;

		size_t kernel_size;

		// dedispersion kernel for all input channels in device memory
    cufftComplex * d_kernels;

    // device scratch memory
    cufftComplex * d_scratch;

    cufftComplex * buf;

    void * work_area;

    size_t work_area_size;

    int auto_allocate;

    int nchan;

    int npol;

    bool fft_configured;

    uint64_t input_stride;

    uint64_t output_stride;

    int npt_fwd;

    int npt_bwd;

    int nbatch;

    unsigned nsamp_overlap;

    unsigned nsamp_step;
    
    unsigned nfilt_pos;

    unsigned nfilt_neg;

#if HAVE_CUFFT_CALLBACKS
		unsigned h_conv_params[2];
#endif

  };
}

#endif


//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2015 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __baseband_cuda_Convolution_h
#define __baseband_cuda_Convolution_h

#include <cufft.h>
#include <config.h>

#include "dsp/Convolution.h"
#include "dsp/LaunchConfig.h"

namespace CUDA
{
  class ConvolutionEngine : public dsp::Convolution::Engine
  {
  public:

    //! Default Constructor
    ConvolutionEngine (cudaStream_t stream);
    ~ConvolutionEngine();

    void set_scratch (void * scratch);

    //! prepare the required attributes for the engine
    void prepare (dsp::Convolution * convolution);

    //! setup the dedispersion kernel from the response
    void setup_kernel (const dsp::Response * response);

    //! configure the singular FFTs
    void setup_singular ();

    //! configure the batched FFTs
    void setup_batched (unsigned nbatch);

#if HAVE_CUFFT_CALLBACKS
    //! setup FFT callbacks
    //void setup_callbacks ();
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

    cufftHandle plan_fwd_batched;

    cufftHandle plan_bwd_batched;

		size_t kernel_size;

		// dedispersion kernel for all input channels in device memory
    cufftComplex * d_kernels;

    // device scratch memory
    cufftComplex * d_scratch;

    cufftComplex * buf;

    void * work_area;

    size_t work_area_size;

    int auto_allocate;

    int npt_fwd;

    int npt_bwd;

    int nbatch;

    unsigned nsamp_overlap;

    unsigned nsamp_step;
    
    unsigned nfilt_pos;

    unsigned nfilt_neg;

#if HAVE_CUFFT_CALLBACKS
		unsigned h_conv_params[5];
#endif

  };
}

#endif


//-*-C++-*-

#include "dsp/Filterbank.h"

#include <cufft.h>
#include <cutil_inline.h>

class FilterbankCUDA : public dsp::Filterbank::Engine
{
  unsigned nstream;

public:

  FilterbankCUDA (unsigned _nstream) { nstream = _nstream; }

  class Stream;

  void setup (unsigned nchan, unsigned bwd_nfft, float* kernel);

  void run ();

  Stream* get_stream (unsigned i);
};

class FilterbankCUDA::Stream : public QuasiMutex::Stream
{
protected:

  //! stream identifier
  cudaStream_t stream;

  //! forward fft plan 
  cufftHandle plan_fwd;
  //! backward fft plan
  cufftHandle plan_bwd;

  //! the backward fft length
  unsigned bwd_nfft;
  //! the number of frequency channels produced by filterbank
  unsigned nchan;

  //! input data in CUDA memory
  float2* d_in;
  //! output data in CUDA memory
  float2* d_out;
  //! convolution kernel in CUDA memory
  float2* d_kernel;

  //! real-to-complex trick arrays in CUDA memory
  float *d_SN, *d_CN;
 
  //! Initializes the attributes that are not shared
  void init ();

  friend class FilterbankCUDA;

  void forward_fft ();
  void realtr ();
  void convolve ();
  void backward_fft ();
  void retrieve ();

public:

  Stream (unsigned nchan, unsigned bwd_nfft, float* kernel);
  Stream (const FilterbankCUDA::Stream&);

  void queue () ;
  void run () ;
  void wait () ;

  class Job {
  public:
    float* in;
    float* out;
  };
};


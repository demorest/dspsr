/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/FilterbankConfig.h"

#if HAVE_CUDA
#include "dsp/MemoryCUDA.h"
#include <cuda_runtime.h>
#endif

#include "CommandLine.h"
#include "RealTimer.h"

#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <math.h>

using namespace std;
using namespace dsp;

class Speed : public Reference::Able
{
public:

  Speed ();

  // parse command line options
  void parseOptions (int argc, char** argv);

  // run the test
  void runTest ();

protected:

  Filterbank::Config config;
  unsigned nloop;
  unsigned niter;
  bool real_to_complex;
  bool do_fwd_fft;
  bool cuda;
};


Speed::Speed ()
{
  niter = 10;
  nloop = 0;
  real_to_complex = false;
  do_fwd_fft = true;
  cuda = false;

  config.set_freq_res( 1024 );
}

int main(int argc, char** argv) try
{
  Speed speed;
  speed.parseOptions (argc, argv);
  speed.runTest ();
  return 0;
}
 catch (Error& error)
   {
     cerr << error << endl;
     return -1;
   }

void Speed::parseOptions (int argc, char** argv)
{
  CommandLine::Menu menu;
  CommandLine::Argument* arg;

  menu.set_help_header ("filterbank_speed - measure Filterbank speed");
  menu.set_version ("filterbank_speed version 1.0");

  arg = menu.add (real_to_complex, 'r');
  arg->set_help ("real-to-complex FFT");

  arg = menu.add (do_fwd_fft, 'b');
  arg->set_help ("do (batched) backward FFTs only");

  arg = menu.add (&config, &Filterbank::Config::set_freq_res, 'n', "nfft");
  arg->set_help ("FFT length");

  arg = menu.add (&config, &Filterbank::Config::set_nchan, 'c', "nchan");
  arg->set_help ("number of channels");

  arg = menu.add (niter, 'N', "niter");
  arg->set_help ("number of iterations");

#if HAVE_CUFFT
  arg = menu.add (cuda, "cuda");
  arg->set_help ("benchmark CUDA");
#endif

  menu.parse (argc, argv);
}

void check_error (const char*);

void Speed::runTest ()
{
  // dsp::Operation::verbose = true;

  unsigned nfloat = config.get_nchan() * config.get_freq_res();
  if (!real_to_complex)
    nfloat *= 2;

  unsigned size = sizeof(float) * nfloat;

  if (!nloop)
  {
    nloop = (1024*1024*256) / size;
    if (nloop > 2000)
      nloop = 2000;
  }

#if HAVE_CUFFT
  if (cuda)
  {
    cudaError_t err = cudaSetDevice (0);
    if (err != cudaSuccess)
      throw Error (InvalidState, "dsp::SingleThread::initialize",
                   "cudaMalloc failed: %s", cudaGetErrorString(err));

    cudaStream_t stream = 0;
    cudaStreamCreate( &stream );

    cerr << "run on GPU" << endl;
    config.set_device( new CUDA::DeviceMemory );
    config.set_stream( stream );
  }
#endif

  dsp::Filterbank* filterbank = config.create();

  dsp::TimeSeries input;
  filterbank->set_input( &input );

  input.set_rate( 1e6 );
  input.set_state( Signal::Analytic );
  input.set_ndim( 2 );
  input.set_input_sample( 0 );

  input.resize( size );
  input.zero();

  dsp::TimeSeries output;
  filterbank->set_output( &output );

  filterbank->prepare();

  RealTimer timer;
  timer.start ();

  for (unsigned i=0; i<nloop; i++)
    filterbank->operate();
    
#if HAVE_CUFFT
  check_error ("CUDA::FilterbankEngine::finish");
#endif

  timer.stop ();
  
  double total_time = timer.get_elapsed();

  double time_us = total_time * 1e6 / (nloop*niter);

  unsigned nfft = config.get_freq_res();
  unsigned nchan = config.get_nchan();

  double log2_nfft = log2(nfft);
  double log2_nchan = log2(nchan);

  double bwd = 2;
  if (nchan == 1)
    bwd = 1;

  double mflops = 5.0 * nfft * nchan * (bwd*log2_nfft + log2_nchan) / time_us;

  cerr << "nchan=" << nchan << " nfft=" << nfft << " time=" << time_us << "us"
    " log2(nfft)=" << log2_nfft << " log2(nchan)=" << log2_nchan << 
    " mflops=" << mflops << endl;

  cout << nchan << " " << nfft << " " << time_us << " "
       << log2_nchan << " " << log2_nfft << " " << mflops << endl;
}



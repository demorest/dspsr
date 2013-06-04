/***************************************************************************
 *
 *   Copyright (C) 2012 by Willem van Straten and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "CommandLine.h"
#include "RealTimer.h"
#include "malloc16.h"

#include "dsp/SpatialFilterbank.h"
#include "dsp/Filterbank.h"
#include "dsp/FilterbankEngine.h"
#include "dsp/Memory.h"

#if HAVE_CUFFT
#include "dsp/SpatialFilterbankCUDA.h"
#include "dsp/MemoryCUDA.h"
#endif

#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <math.h>

using namespace std;

class Speed : public Reference::Able
{
public:

  Speed ();

  // parse command line options
  void parseOptions (int argc, char** argv);

  // run the test
  void runTest ();

protected:

  unsigned nx;
  unsigned ny;
  unsigned nbatch;
  unsigned ram;
  unsigned nloop;
  unsigned niter;
  bool real_to_complex;
};


Speed::Speed ()
{
  niter = 10;
  nx = 256;
  ny = 128;
  nbatch = 1;
  ram = 256;
  real_to_complex = false;
}

int main(int argc, char** argv) try
{
  //dsp::set_verbosity (3);
  
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

  menu.set_help_header ("filterbank_speed_2d - measure Batched 2D Filterbank speed");
  menu.set_version ("filterbank_speed_2d version 1.0");

  arg = menu.add (real_to_complex, 'r');
  arg->set_help ("real-to-complex FFT");

  arg = menu.add (nx, 'x', "nx");
  arg->set_help ("FFT length x dimension");

  arg = menu.add (ny, 'y', "ny");
  arg->set_help ("FFT length y dimension");

  arg = menu.add (nbatch, 'b', "nbatch");
  arg->set_help ("number of batches");

  arg = menu.add (niter, 'N', "niter");
  arg->set_help ("number of iterations");

  arg = menu.add (ram, 'R', "ram");
  arg->set_help ("MB of RAM to use");


  menu.parse (argc, argv);
}

double order (unsigned nfft)
{
  return nfft * log2 (nfft);
}

void Speed::runTest ()
{
  // determine number of floats required for input
  unsigned nfloat = nx * ny * nbatch;
  unsigned nbit = sizeof(float) * 8;
  unsigned npol = 1;
  unsigned ndim;
  unsigned nsamp_fft;
  Signal::State state;

  if (real_to_complex)
  {
    ndim = 1;
    nsamp_fft = 2 * nfloat;
    state = Signal::Nyquist;
  }
  else
  {
    ndim = 2;
    nsamp_fft = nfloat;
    state = Signal::Analytic;
    //nfloat *= 2;
  }
  ram *= 1024*1024;

  unsigned bytes_per_sample = (nbit / 8) * ndim * npol;
  uint64_t ndat = ram / bytes_per_sample;
  uint64_t output_ndat = ndat / (nx * ny);

  cerr << "Speed::runTest nbit=" << nbit << endl;
  cerr << "Speed::runTest ndim=" << ndim << endl;
  cerr << "Speed::runTest npol=" << npol << endl;
  cerr << "Speed::runTest bytes_per_sample=" << bytes_per_sample << endl;
  cerr << "Speed::runTest ram=" << ram << endl;
  cerr << "Speed::runTest ndat=" << ndat << endl;

  cerr << "Speed::runTest nx [antenna]=" << nx << endl;
  cerr << "Speed::runTest ny [chan]=" << ny << endl;
  cerr << "Speed::runTest output_ndat=" << output_ndat << endl;
  cerr << "Speed::runTest nfloat=" << nfloat << endl;

  //unsigned size = sizeof(float) * nfloat;

  dsp::Filterbank::Engine* engine = 0;
  dsp::Memory* memory = 0;

#if HAVE_CUFFT
  cudaStream_t stream;
  cudaStreamCreate( &stream );
  engine = new CUDA::SpatialFilterbankEngine (stream);
  memory = new CUDA::DeviceMemory;
#endif

  if (!memory)
    memory = new dsp::Memory;

  if (!engine)
    throw Error (InvalidState, "Speed::runTest",
                 "engine not set");

  dsp::TimeSeries input;
  dsp::TimeSeries output;

  input.set_memory (memory);
  output.set_memory (memory);

  input.set_ndim ( ndim );
  input.set_nbit ( nbit );
  input.set_npol ( npol );
  input.set_state ( state );
  input.resize ( ndat );

  output.set_npol (input.get_npol());
  output.set_nchan (ny);
  output.set_ndim (2);
  output.set_state (Signal::Analytic);
  output.set_order (dsp::TimeSeries::OrderTFP);

  dsp::SpatialFilterbank temp;
  temp.set_nchan (ny);
  temp.set_frequency_resolution (nx);
  temp.set_input (&input);
  temp.set_output (&output);
  temp.prepare ();

  // now configure engine
  engine->setup (&temp);

  unsigned nfilt_pos = 0;
  unsigned freq_res = nx;
  unsigned nkeep = freq_res;

  cerr << "Speed::runTest engine->configure(" << ny << ", " << nfilt_pos << ", " << freq_res << ", " << nkeep << ")" << endl;
  engine->configure (ny, nfilt_pos, freq_res, nkeep);

  uint64_t npart = ndat / nsamp_fft;
  uint64_t in_step = nsamp_fft * ndim;
  uint64_t out_step = nfloat * npol * 2;

  double total_time = 0;

  for (unsigned j=0; j<niter; j++)
  {
    RealTimer timer;
    timer.start ();

    engine->perform (&input, &output, npart, in_step, out_step);

    engine->finish ();

    timer.stop ();

    total_time += timer.get_elapsed();
  }

  double time_us = total_time * 1e6 / (niter*npart*npol);
  double log2_nxny = log2(nx* ny);
  double bwd = 2;

  double mflops = 5.0 * nx * ny * log2_nxny / time_us;

  cerr << "ny=" << ny << " nx=" << nx << " time=" << time_us << "us"
    " log2(nxny)=" << log2_nxny << " mflops=" << mflops << endl;

  cout << ny << " " << nx << " " << time_us << " "
       << log2_nxny << " " << mflops << endl;
}


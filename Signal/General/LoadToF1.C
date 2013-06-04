/***************************************************************************
 *
 *   Copyright (C) 2007-2013 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/LoadToF1.h"

#include "dsp/IOManager.h"
#include "dsp/Unpacker.h"
#include "dsp/Scratch.h"

#include "dsp/SpatialFilterbank.h"
#include "dsp/TPFFilterbank.h"
#include "dsp/Filterbank.h"
#include "dsp/FourierDigitizer.h"

#if HAVE_CUDA
#include "dsp/SpatialFilterbankCUDA.h"
#include "dsp/TPFFilterbankCUDA.h"
#include "dsp/FourierDigitizerCUDA.h"
#include "dsp/DetectionCUDA.h"
#include "dsp/AccumulationCUDA.h"
#include "dsp/OptimalFilterbank.h"
#include "dsp/TransferCUDA.h"
#include "dsp/TransferBitSeriesCUDA.h"
#include "dsp/MemoryCUDA.h"
#endif

//#include "dsp/FourierDigitizer.h"
//#include "dsp/PSRDadaOutputStream.h"

using namespace std;

bool dsp::LoadToF::verbose = false;

static void* const undefined_stream = (void *) -1;

dsp::LoadToF::LoadToF (Config* configuration)
{
  set_configuration (configuration);
}

//! Run through the data
void dsp::LoadToF::set_configuration (Config* configuration)
{
  SingleThread::set_configuration (configuration);
  config = configuration;
}

dsp::LoadToF::Config::Config()
{
  can_cuda = true;
  can_thread = true;

  // block size in MB
  block_size = 2.0;
  nbatch = 1;
  acc_len = 125;

  order = dsp::TimeSeries::OrderTFP;
  std::cerr << "dsp::LoadToF::Config::Config setting order=TFP" << endl;
 
  filterbank.set_nchan(0);
  filterbank.set_freq_res(512);
  filterbank.set_convolve_when(Filterbank::Config::Never);

  nbits = 8;

  // by default, time series weights are not used
  weighted_time_series = false;
}

void dsp::LoadToF::Config::set_quiet ()
{
  SingleThread::Config::set_quiet();
  LoadToF::verbose = false;
}

void dsp::LoadToF::Config::set_verbose ()
{
  SingleThread::Config::set_verbose();
  LoadToF::verbose = true;
}

void dsp::LoadToF::Config::set_very_verbose ()
{
  SingleThread::Config::set_very_verbose();
  LoadToF::verbose = true;
}

void dsp::LoadToF::construct () try
{

#if HAVE_CUDA
  bool run_on_gpu = thread_id < config->get_cuda_ndevice();
  cudaStream_t stream = reinterpret_cast<cudaStream_t>( gpu_stream );
  if (verbose)
    cerr <<"dsp::LoadToF::construct stream=" << stream << endl;
#endif

  // TODO parameterise this [number of antenna's will require some sort of padding on the input read...
  unsigned frequency_resolution = config->filterbank.get_freq_res ();
  frequency_resolution = 512;

  if (verbose)
    cerr << "dsp::LoadToF::construct frequency_resolution=" << frequency_resolution << endl;
  /*
    The following lines "wire up" the signal path, using containers
    to communicate the data between operations.
  */

  // set up for optimal memory usage pattern
  Unpacker* unpacker = manager->get_unpacker();

  if (unpacker->get_order_supported (config->order))
    unpacker->set_output_order (config->order);

  // get basic information about the observation
  Observation* obs = manager->get_info();
  const unsigned nchan = obs->get_nchan ();
  const unsigned npol = obs->get_npol ();
  const unsigned ndim = obs->get_ndim ();

  if (verbose)
  {
    cerr << "Source = " << obs->get_source() << endl;
    cerr << "Frequency = " << obs->get_centre_frequency() << endl;
    cerr << "Bandwidth = " << obs->get_bandwidth() << endl;
    cerr << "Sampling rate = " << obs->get_rate() << endl;
  }

  // the unpacked input will occupy nbytes_per_sample
  double nbytes_per_sample = sizeof(float) * nchan * npol * ndim;

  double MB = 1024.0 * 1024.0;
  uint64_t nsample = uint64_t( config->block_size*MB / nbytes_per_sample );

  if (verbose)
    cerr << "loadToF: block_size=" << config->block_size << " MB "
      "(" << nsample << " samp)" << endl;

  uint64_t sample_multiple = frequency_resolution * config->filterbank.get_nchan () * config->acc_len;
  if (nsample % sample_multiple != 0)
  {
    uint64_t nblocks = nsample / sample_multiple;
    nsample = nblocks * sample_multiple;
  } 

  manager->set_block_size( nsample );
 
  // new storage for filterbank output (must be out of place)
  TimeSeries * channelised = new_time_series ();

  if (npol != 1)
    throw Error (InvalidState, "loadToF::construct",
      "currently only 1 polarisation supported");

  if (config->filterbank.get_nchan() <= 1)
    throw Error (InvalidState, "loadToF::construct", 
      "not sensible to use <= 1 Filterbank channel");

  if (!filterbank)
  {
    filterbank = new SpatialFilterbank;
  }

  if (!config->input_buffering)
    filterbank->set_buffering_policy (NULL);

  filterbank->set_input (unpacked);
  filterbank->set_output (channelised);
  filterbank->set_nchan (config->filterbank.get_nchan ());

  if (frequency_resolution)
  {
    cerr << "loadToF: setting frequnecy resolution" << endl;
    filterbank->set_frequency_resolution (frequency_resolution);
  }

#if HAVE_CUDA
  if (run_on_gpu)
  {
    filterbank->set_engine (new CUDA::SpatialFilterbankEngine (stream));
    channelised->set_memory (device_memory);
  }
#endif

  operations.push_back (filterbank.get());

  if (!detect)
    detect = new Detection;

  TimeSeries* detected = new_time_series();
  detected->set_order(dsp::TimeSeries::OrderTFP);

  detect->set_input (channelised);
  detect->set_output (detected);

#if HAVE_CUDA
  if (run_on_gpu)
  {
    detect->set_engine (new CUDA::DetectionEngine(stream));
    detected->set_memory (device_memory);
  }
#endif
  if (manager->get_info()->get_npol() == 1)
  {
    detect->set_output_ndim (1);
    detect->set_output_state(Signal::Intensity);
  }
  else
    throw Error (InvalidState, "loadToF::construct",
      "no support for more than 1 polatisation");

  operations.push_back (detect.get());

  // Accumulate 
  if (!accumulate)
    accumulate = new Accumulation (config->acc_len);

  accumulate->set_input (detected);

  // CPU based timeseries
  TimeSeries* integrated_cpu = new_time_series();
  integrated_cpu->set_order (dsp::TimeSeries::OrderTFP);

#if HAVE_CUDA
  TimeSeries* integrated_gpu = new_time_series();
  integrated_gpu->set_order (dsp::TimeSeries::OrderTFP);

  if (run_on_gpu)
  {
    accumulate->set_engine (new CUDA::AccumulationEngine(stream));
    integrated_gpu->set_memory (device_memory);
    accumulate->set_output (integrated_gpu);
    integrated_gpu->set_state (Signal::Intensity);
  }
  else
  {
    accumulate->set_output (integrated_cpu);
    integrated_cpu->set_state (Signal::Intensity);
  }
#else
  accumulate->set_output (integrated_cpu);
  integrated_cpu->set_state (Signal::Intensity);
#endif
  operations.push_back (accumulate.get());

  //! Device to Host transfer

#if HAVE_CUDA
  if (run_on_gpu)
  {
    // transfer the integrated totals back from the GPU
    if (!copyback)
      copyback = new TransferCUDA (stream);
    copyback->set_pre_sync (false);
    copyback->set_post_sync (true);
    copyback->set_kind (cudaMemcpyDeviceToHost);
    copyback->set_input (integrated_gpu);
    copyback->set_output (integrated_cpu);
    integrated_cpu->set_memory (new CUDA::PinnedMemory);

    operations.push_back (copyback.get());
  }
#endif

/*
  // TODO rescale operation to convert data from 32-bit to 8-bit
  if (verbose)
    cerr << "dspF: creating output bitseries container" << endl;

  BitSeries* bitseries = new BitSeries;

  if (verbose)
    cerr << "dspF: creating digitizer [" << config->nbits << " bits]" << endl;

  FourierDigitizer* digitizer = new FourierDigitizer;
  digitizer->set_nbit (config->nbits);
  digitizer->set_input (spectra);
  digitizer->set_output (bitseries);

#if HAVE_CUDA
  if (run_on_gpu)
  {
    digitizer->set_engine (new CUDA::FourierDigitizerEngine (stream));
    bitseries->set_memory (device_memory);
  }
#endif

  operations.push_back (digitizer);
*/
  // TODO transfer operation to copy data back from GPU to CPU RAM

  // TODO improvement on above to allow copy into raw PSRDADA buffers

/*
  if (verbose)
    cerr << "dspF: creating PSRDADA output stream" << endl;

  DADAOutputStream * outputStream = new DADAOutputStream (config->output_key);
  outputStream->set_input (bitseries);
  operations.push_back( outputStream );
*/

}
catch (Error& error)
{
  throw error += "dsp::LoadToF::construct";
}

void dsp::LoadToF::finalize () try
{
  SingleThread::finalize();

  // Check that block size is sufficient for the filterbanks,
  // increase it if not.
  if (verbose)
    cerr << "dspF: filterbank minimum samples = " 
      << filterbank->get_minimum_samples() 
      << endl;

  if (filterbank->get_minimum_samples() > 
      manager->get_input()->get_block_size())
  {
    cerr << "dspF: increasing data block size from " 
      << manager->get_input()->get_block_size()
      << " to " << filterbank->get_minimum_samples() 
      << " samples" << endl;
    manager->set_block_size( filterbank->get_minimum_samples() );
  }

}
catch (Error& error)
{
  throw error += "dsp::LoadToF::finalize";
}


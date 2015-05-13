/***************************************************************************
 *
 *   Copyright (C) 2015 by Matthew Kerr
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/LoadToFITS.h"

#include "dsp/IOManager.h"
#include "dsp/Unpacker.h"

#include "dsp/TFPFilterbank.h"
#include "dsp/Filterbank.h"
#include "dsp/Detection.h"

#include "dsp/SampleDelay.h"
#include "dsp/DedispersionSampleDelay.h"

#include "dsp/FScrunch.h"
#include "dsp/TScrunch.h"
#include "dsp/PScrunch.h"
#include "dsp/PolnSelect.h"
#include "dsp/PolnReshape.h"

#include "dsp/Rescale.h"

#include "dsp/FITSDigitizer.h"
#include "dsp/FITSOutputFile.h"

#if HAVE_CUDA
#include "dsp/FilterbankCUDA.h"
#include "dsp/OptimalFilterbank.h"
#include "dsp/TransferCUDA.h"
#include "dsp/DetectionCUDA.h"
#include "dsp/FScrunchCUDA.h"
#include "dsp/MemoryCUDA.h"
#endif


using namespace std;

bool dsp::LoadToFITS::verbose = false;

static void* const undefined_stream = (void *) -1;

dsp::LoadToFITS::LoadToFITS (Config* configuration)
{
  kernel = NULL;
  set_configuration (configuration);
}

//! Run through the data
void dsp::LoadToFITS::set_configuration (Config* configuration)
{
  SingleThread::set_configuration (configuration);
  config = configuration;
}

dsp::LoadToFITS::Config::Config()
{
  can_cuda = true;
  can_thread = true;

  // block size in MB
  block_size = 2.0;

  order = dsp::TimeSeries::OrderTFP;
 
  filterbank.set_nchan(0);
  filterbank.set_freq_res(0);
  filterbank.set_convolve_when(Filterbank::Config::Never);

  dispersion_measure = 0;
  dedisperse = false;
  coherent_dedisp = false;

  tscrunch_factor = 0;
  fscrunch_factor = 0;

  rescale_seconds = -1;
  rescale_constant = false;

  nbits = 2;

  npol = 4;

  nsblk = 2048;
  tsamp = 64e-6;

  // by default, time series weights are not used
  weighted_time_series = false;
}

void dsp::LoadToFITS::Config::set_quiet ()
{
  SingleThread::Config::set_quiet();
  LoadToFITS::verbose = false;
}

void dsp::LoadToFITS::Config::set_verbose ()
{
  SingleThread::Config::set_verbose();
  LoadToFITS::verbose = true;
}

void dsp::LoadToFITS::Config::set_very_verbose ()
{
  SingleThread::Config::set_very_verbose();
  LoadToFITS::verbose = 3;
  FITSOutputFile::verbose = 3;
}

void dsp::LoadToFITS::construct () try
{
  SingleThread::construct ();

#if HAVE_CUDA
  bool run_on_gpu = thread_id < config->get_cuda_ndevice();
  cudaStream_t stream = reinterpret_cast<cudaStream_t>( gpu_stream );
#endif

  /*
    The following lines "wire up" the signal path, using containers
    to communicate the data between operations.
  */

  // set up for optimal memory usage pattern

  Unpacker* unpacker = manager->get_unpacker();
  
  if (!config->dedisperse && unpacker->get_order_supported (config->order))
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

  obs->set_dispersion_measure( config->dispersion_measure );

  // for now, handle spectral leakage AND integrating up a filterbank sample
  // by using the frequency resolution aspect of Filterbank
  double samp_per_fb = config->tsamp*obs->get_rate();
  if (obs->get_state()==Signal::Nyquist)
    samp_per_fb *= 0.5;
  unsigned res_factor = round(samp_per_fb/config->filterbank.get_nchan());
  //if (verbose)
    cerr << "dsp::LoadToFITS::construct " <<"tsamp="<<config->tsamp<<" rate="<<obs->get_rate() << " so increasing spectral resolution by "<<res_factor << endl;

  uint64_t nsample = round(samp_per_fb * 4096);


  // the unpacked input will occupy nbytes_per_sample
  double nbytes_per_sample = sizeof(float) * nchan * npol * ndim;

  double MB = 1024.0 * 1024.0;
  //uint64_t nsample = uint64_t( config->block_size*MB / nbytes_per_sample );

  if (verbose)
    //cerr << "digifits: block_size=" << config->block_size << " MB "
    //  "(" << nsample << " samp)" << endl;
    cerr << "digifits: block_size=" << (nbytes_per_sample*nsample)/MB << " MB " << "(" << nsample << " samp)" << endl;

  // TODO -- set block size more adequately
  manager->set_block_size( nsample );

  bool do_pscrunch = false;
  //bool do_pscrunch = (obs->get_npol() > 1) && (config->poln_select < 0);

  //cerr << "tsamp="<<config->tsamp<<" rate="<<obs->get_rate()<<endl;
  TimeSeries* timeseries = unpacked;

# if HAVE_CUDA
  if (run_on_gpu)
  {
    TransferCUDA* transfer = new TransferCUDA (stream);
    transfer->set_kind (cudaMemcpyDeviceToHost);
    transfer->set_input( timeseries );
    transfer->set_output( timeseries = new_TimeSeries() );
    operations.push_back (transfer);
    run_on_gpu = false;
  }
#endif

  if (!obs->get_detected())
  {
    bool do_detection = false;

    config->coherent_dedisp = 
      (config->filterbank.get_convolve_when() == Filterbank::Config::During)
      && (config->dispersion_measure != 0.0);

    if ( config->filterbank.get_nchan() )
    {
      if (verbose)
	      cerr << "digifits: creating " << config->filterbank.get_nchan()
	           << " channel filterbank" << endl;

      if ( config->coherent_dedisp )
      {
	      cerr << "digifits: using coherent dedispersion" << endl;

        kernel = new Dedispersion;

        if (config->filterbank.get_freq_res())
          kernel->set_frequency_resolution (config->filterbank.get_freq_res());

        kernel->set_dispersion_measure( config->dispersion_measure );

        // TODO other FFT length/etc options as implemented in LoadToFold1?

      }

      //if ( config->filterbank.get_freq_res() || config->coherent_dedisp )
      if (true)
      {
	      cerr << "digifits: using convolving filterbank" << endl;
# if HAVE_CUDA
        if (run_on_gpu)
        {
          timeseries->set_memory (device_memory);
          config->filterbank.set_device ( device_memory.ptr() );
          config->filterbank.set_stream ( gpu_stream );
        }
#endif

	      filterbank = config->filterbank.create ();

	      filterbank->set_nchan( config->filterbank.get_nchan()*res_factor );
	      filterbank->set_input( timeseries );
        filterbank->set_output( timeseries = new_TimeSeries() );
        //filterbank->set_frequency_resolution (40);
# if HAVE_CUDA
        if (run_on_gpu)
          timeseries->set_memory (device_memory);
#endif

        if (kernel)
          filterbank->set_response( kernel );

	      //filterbank->set_frequency_resolution ( 
        //    config->filterbank.get_freq_res() );

	      operations.push_back( filterbank.get() );
	      do_detection = true;

      }
      else
      {
	      filterbank = new TFPFilterbank;

	      filterbank->set_nchan( config->filterbank.get_nchan() );
        filterbank->set_input( timeseries );
        filterbank->set_output( timeseries = new_TimeSeries() );

	      operations.push_back( filterbank.get() );
        do_detection = true;
      }
    }



    if (do_detection)
    {
      if (verbose)
	      cerr << "digifits: creating detection operation" << endl;
      
      Detection* detection = new Detection;

      switch (config->npol)
      {
        case 1:
          detection->set_output_state (Signal::Intensity);
          break;
        case 2:
          detection->set_output_state (Signal::PPQQ);
          break;
        case 4:
          detection->set_output_state (Signal::Coherence);
          detection->set_output_ndim(2);
# if HAVE_CUDA
          //if (run_on_gpu)
          if (false)
          {
            detection->set_engine (new CUDA::DetectionEngine(stream) );
            detection->set_output_ndim (2);
          }
#endif
          break;
        default:
          throw Error(InvalidParam,"dsp::LoadToFITS::construct",
              "invalid polarization specified");
      }

      detection->set_input( timeseries );
      detection->set_output( timeseries );

      // detection will do pscrunch
      do_pscrunch = false;

      operations.push_back( detection );
    }
  }

  //if ( config->dedisperse )
  if (false)
  {
    if (verbose)
      cerr << "digifits: removing dispserion delays" << endl;

    SampleDelay* delay = new SampleDelay;

    delay->set_input (timeseries);
    delay->set_output (timeseries);
    delay->set_function (new Dedispersion::SampleDelay);

    operations.push_back( delay );
  }

  FScrunch* fscrunch = new FScrunch;
  
  fscrunch->set_factor( res_factor );
  fscrunch->set_input( timeseries );
//# if HAVE_CUDA
//  if (run_on_gpu)
//  {
//    fscrunch->set_engine ( new CUDA::FScrunchEngine(stream) );
//    timeseries = new_TimeSeries();
//    timeseries->set_memory (device_memory);
//  }
//#endif
  fscrunch->set_output( timeseries );

  operations.push_back( fscrunch );

  /*
# if HAVE_CUDA
  if (run_on_gpu)
  {
    TransferCUDA* transfer = new TransferCUDA (stream);
    transfer->set_kind (cudaMemcpyDeviceToHost);
    transfer->set_input( timeseries );
    transfer->set_output( timeseries = new_TimeSeries() );
    operations.push_back (transfer);

    PolnReshape* reshape = new PolnReshape;
    reshape->set_state ( Signal::Coherence );
    reshape->set_input (timeseries );
    reshape->set_output ( timeseries = new_TimeSeries() );
    operations.push_back(reshape);
  }
#endif
*/
    PolnReshape* reshape = new PolnReshape;
    reshape->set_state ( Signal::Coherence );
    reshape->set_input (timeseries );
    reshape->set_output ( timeseries = new_TimeSeries() );
    operations.push_back(reshape);

  if ( config->tscrunch_factor )
  {
    TScrunch* tscrunch = new TScrunch;
    
    tscrunch->set_factor( config->tscrunch_factor );
    tscrunch->set_input( timeseries );
    tscrunch->set_output( timeseries );

    operations.push_back( tscrunch );
  }
  
  // PSRFITS allows us to save the reference spectrum in each output block
  // "subint", so we can take advantage of this to store the exect
  // reference spectrum for later use if we select a rescale time cons.
  // exactly equal to nsblk; do this by default unless the user manually
  // specifies a time constant
  if (verbose)
    cerr << "digifits: creating rescale transformation" << endl;

  Rescale* rescale = new Rescale;
  rescale->set_input (timeseries);
  rescale->set_output (timeseries);
  if (config->rescale_seconds) 
    rescale->set_interval_seconds (config->rescale_seconds);
  else
    rescale->set_interval_samples (config->nsblk);
  rescale->set_constant (config->rescale_constant);

  operations.push_back( rescale );

  if (do_pscrunch)
  {
    if (verbose)
      cerr << "digifits: creating pscrunch transformation" << endl;

    PScrunch* pscrunch = new PScrunch;
    pscrunch->set_input (timeseries);
    pscrunch->set_output (timeseries);

    operations.push_back( pscrunch );
  }

  if (verbose)
    cerr << "digifits: creating output bitseries container" << endl;

  BitSeries* bitseries = new BitSeries;

  if (verbose)
    cerr << "digifits: creating PSRFITS digitizer with nbit="<<config->nbits << endl;

  FITSDigitizer* digitizer = new FITSDigitizer (config->nbits);
  digitizer->set_input (timeseries);
  digitizer->set_output (bitseries);

  operations.push_back( digitizer );

  if (verbose)
    cerr << "digifits: creating PSRFITS output file" << endl;

  const char* output_filename = 0;
  if (!config->output_filename.empty())
    output_filename = config->output_filename.c_str();

  // temporary
  std::string fname = "/tmp/test.sf";
  output_filename = fname.c_str();

  FITSOutputFile* outputfile = new FITSOutputFile (output_filename);
  outputfile->set_nsblk (config->nsblk);
  outputfile->set_nbit (config->nbits);
  outputFile = outputfile;
  outputFile->set_input (bitseries);

  operations.push_back( outputFile.get() );

  // add a callback for the PSRFITS reference spectrum
  rescale->update.connect (
      dynamic_cast<FITSOutputFile*> (outputFile.get()), 
      &FITSOutputFile::set_reference_spectrum);
}
catch (Error& error)
{
  throw error += "dsp::LoadToFITS::construct";
}

void dsp::LoadToFITS::prepare () try
{
  SingleThread::prepare();
  
  // TODO -- set an optimal block size for search mode

  // Check that block size is sufficient for the filterbanks,
  // increase it if not.
  if (filterbank && verbose)
    cerr << "digifits: filterbank minimum samples = " 
      << filterbank->get_minimum_samples() 
      << endl;

  if (filterbank) {
    if (filterbank->get_minimum_samples() > 
        manager->get_input()->get_block_size())
    {
      cerr << "digifits: increasing data block size from " 
        << manager->get_input()->get_block_size()
        << " to " << filterbank->get_minimum_samples() 
        << " samples" << endl;
      manager->set_block_size( filterbank->get_minimum_samples() );
    }
  }

}
catch (Error& error)
{
  throw error += "dsp::LoadToFITS::prepare";
}

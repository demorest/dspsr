/***************************************************************************
 *
 *   Copyright (C) 2007-2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/LoadToFold1.h"
#include "dsp/LoadToFoldConfig.h"

#include "dsp/SignalPath.h"
#include "dsp/IOManager.h"
#include "dsp/Input.h"
#include "dsp/Scratch.h"
#include "dsp/File.h"

#include "dsp/ExcisionUnpacker.h"
#include "dsp/WeightedTimeSeries.h"

#include "dsp/ResponseProduct.h"
#include "dsp/DedispersionSampleDelay.h"
#include "dsp/RFIFilter.h"
#include "dsp/PolnCalibration.h"

#include "dsp/Filterbank.h"
#include "dsp/SKFilterbank.h"
#include "dsp/SKDetector.h"
#include "dsp/SKMasker.h"
#include "dsp/OptimalFFT.h"
#include "dsp/Resize.h"

#if HAVE_CUDA
#include "dsp/FilterbankCUDA.h"
#include "dsp/OptimalFilterbank.h"
#include "dsp/TransferCUDA.h"
#include "dsp/TransferBitSeriesCUDA.h"
#include "dsp/DetectionCUDA.h"
#include "dsp/FoldCUDA.h"
#include "dsp/MemoryCUDA.h"
#include "dsp/SKMaskerCUDA.h"
#endif

#include "dsp/SampleDelay.h"
#include "dsp/PhaseLockedFilterbank.h"
#include "dsp/Detection.h"
#include "dsp/FourthMoment.h"
#include "dsp/Stats.h"

#include "dsp/Fold.h"
#include "dsp/Subint.h"
#include "dsp/PhaseSeries.h"
#include "dsp/OperationThread.h"

#include "dsp/CyclicFold.h"

#include "dsp/Archiver.h"
#include "dsp/ObservationChange.h"
#include "dsp/Dump.h"

#include "Pulsar/Archive.h"
#include "Pulsar/TextParameters.h"
#include "Pulsar/SimplePredictor.h"

#include "Error.h"

#include <assert.h>

using namespace std;

static void* const undefined_stream = (void *) -1;

dsp::LoadToFold::LoadToFold (Config* configuration)
{
  manage_archiver = true;
  set_configuration (configuration);
}

dsp::LoadToFold::~LoadToFold ()
{
}

//! Run through the data
void dsp::LoadToFold::set_configuration (Config* configuration)
{
  SingleThread::set_configuration (configuration);
  config = configuration;
}


template<typename T>
unsigned count (const std::vector<T>& data, T element)
{
  unsigned c = 0;
  for (unsigned i=0; i<data.size(); i++)
    if (data[i] == element)
      c ++;
  return c;
}

void dsp::LoadToFold::construct () try
{

#if HAVE_CUDA
  bool run_on_gpu = thread_id < config->get_cuda_ndevice();
  cudaStream_t stream = reinterpret_cast<cudaStream_t>( gpu_stream );
#endif

  if (manager->get_info()->get_detected())
  {
    Unpacker* unpacker = manager->get_unpacker();

    // detected data is handled much more efficiently in TFP order
    if (unpacker->get_order_supported (TimeSeries::OrderTFP))
      unpacker->set_output_order (TimeSeries::OrderTFP);

    config->coherent_dedispersion = false;
    prepare_interchan (unpacked);
    prepare_fold (unpacked);
    return;
  }

  bool report_vitals = thread_id==0 && config->report_vitals;

  if (manager->get_info()->get_type() != Signal::Pulsar)
  {
    // the kernel gets messed up by DM=0 sources, like PolnCal
    if (report_vitals)
      cerr << "Disabling coherent dedispersion of non-pulsar signal" << endl;
    config->coherent_dedispersion = false;
  }

  // the data are not detected, so set up phase coherent reduction path
  unsigned frequency_resolution = config->filterbank.get_freq_res ();

  if (config->coherent_dedispersion)
  {
    if (!kernel)
      kernel = new Dedispersion;

    if (frequency_resolution)
    {
      if (report_vitals)
	cerr << "dspsr: setting filter length to " << frequency_resolution << endl;
      kernel->set_frequency_resolution (frequency_resolution);
    }

    if (config->times_minimum_nfft)
    {
      if (report_vitals)
        cerr << "dspsr: setting filter length to minimum times " 
             << config->times_minimum_nfft << endl;
      kernel->set_times_minimum_nfft (config->times_minimum_nfft);
    }

    if (config->nsmear)
    {
      if (report_vitals)
        cerr << "dspsr: setting smearing to " << config->nsmear << endl;
      kernel->set_smearing_samples (config->nsmear);
    }

    if (config->use_fft_bench)
    {
      if (report_vitals)
        cerr << "dspsr: using benchmarks to choose optimal FFT length" << endl;

#if HAVE_CUDA
      if (run_on_gpu)
        kernel->set_optimal_fft( new OptimalFilterbank("CUDA") );
      else
#endif
        kernel->set_optimal_fft( new OptimalFFT );
    }
  }
  else
    kernel = 0;


  if (!config->single_pulse && !passband)
    passband = new Response;

  Response* response = kernel.ptr();

  if (config->zap_rfi)
  {
    if (!rfi_filter)
      rfi_filter = new RFIFilter;

    rfi_filter->set_input (manager);

    response = rfi_filter;

    if (kernel)
    {
      if (!response_product)
        response_product = new ResponseProduct;

      response_product->add_response (kernel);
      response_product->add_response (rfi_filter);

      response = response_product;
    }

  }

  if (!config->calibrator_database_filename.empty())
  {
    dsp::PolnCalibration* polcal = new PolnCalibration;
   
    polcal-> set_database_filename (config->calibrator_database_filename);

    if (kernel)
    {
      if (!response_product)
        response_product = new ResponseProduct;
    
      response_product->add_response (polcal);
      response_product->add_response (kernel);
      
      response_product->set_copy_index (0);
      response_product->set_match_index (1);
      
      response = response_product;
    }
  }

  // only the Filterbank must be out-of-place
  TimeSeries* convolved = unpacked;

  TimeSeries* skoutput = 0;
  BitSeries * skzapmask = 0;
  Reference::To<OperationThread> skthread;

  if (config->sk_zap)
  {
    // put the SK signal path into a separate thread
    skthread = new OperationThread();

    TimeSeries* skfilterbank_input = unpacked;

#if HAVE_CUDA
    if (run_on_gpu) 
    {
      Unpacker* unpack_on_cpu = 0;
      unpack_on_cpu = manager->get_unpacker()->clone();
      unpack_on_cpu->set_device (Memory::get_manager());

      unpack_on_cpu->set_input( manager->get_unpacker()->get_input() );
      unpack_on_cpu->set_output( skfilterbank_input = new_time_series() );

      skthread->append_operation( unpack_on_cpu );
      manager->set_post_load_operation( skthread.get() );
    }
#endif

    skoutput = new_time_series ();

    // Spectral Kurtosis filterbank constructor
    if (!skfilterbank)
      skfilterbank = new SKFilterbank (config->sk_nthreads);

    if (!config->input_buffering)
      skfilterbank->set_buffering_policy (NULL);

    skfilterbank->set_input ( skfilterbank_input );

    skfilterbank->set_output ( skoutput );
    skfilterbank->set_nchan ( config->filterbank.get_nchan() );
    skfilterbank->set_M ( config->sk_m );

    // SKFB also maintains trscunched SK stats
    TimeSeries* skoutput_tscr = new_time_series();

    skfilterbank->set_output_tscr (skoutput_tscr);

    skthread->append_operation (skfilterbank.get());

    // SK Mask Generator
    skzapmask = new BitSeries;
    skzapmask->set_nbit (8);
    skzapmask->set_npol (1);
    skzapmask->set_nchan (config->filterbank.get_nchan());

    SKDetector * skdetector = new SKDetector;
    skdetector->set_input (skoutput);
    skdetector->set_input_tscr (skoutput_tscr);
    skdetector->set_output (skzapmask);

    skdetector->set_thresholds (config->sk_m, config->sk_std_devs);
    if (config->sk_chan_start > 0 && config->sk_chan_end < config->filterbank.get_nchan())
      skdetector->set_channel_range (config->sk_chan_start, config->sk_chan_end);
    skdetector->set_options (config->sk_no_fscr, config->sk_no_tscr, config->sk_no_ft); 

    skthread->append_operation (skdetector);

#if HAVE_CUDA
    if (!run_on_gpu)
#endif
    {
      operations.push_back (skthread.get());
      OperationThread::Wait * skthread_wait = skthread->get_wait();
      operations.push_back (skthread_wait);
    }

    // since the blocksize is artificially increased for the SKFB,
    // we must return it to the required size for the SKFB
    if (!skresize)
      skresize = new Resize;
    
    skresize->set_input(unpacked);
    skresize->set_output(unpacked);
    operations.push_back (skresize.get());

  }

  if (config->filterbank.get_nchan() > 1)
  {
    // new storage for filterbank output (must be out-of-place)
    convolved = new_time_series ();

    // software filterbank constructor
    if (!filterbank)
      filterbank = new Filterbank;

    if (!config->input_buffering)
      filterbank->set_buffering_policy (NULL);

    filterbank->set_input (unpacked);
    filterbank->set_output (convolved);
    filterbank->set_nchan (config->filterbank.get_nchan ());
    
    if (config->filterbank.get_convolve_when() == Filterbank::Config::During)
    {
      filterbank->set_response (response);
      if (!config->single_pulse)
        filterbank->set_passband (passband);
    }

    if (frequency_resolution)
      filterbank->set_frequency_resolution (frequency_resolution);

    // Get order of operations correct
    if (!config->filterbank.get_convolve_when() == Filterbank::Config::Before)
      operations.push_back (filterbank.get());

#if HAVE_CUDA
    if (run_on_gpu)
    {
      filterbank->set_engine (new CUDA::FilterbankEngine (stream));
      convolved->set_memory (device_memory);

      Scratch* gpu_scratch = new Scratch;
      gpu_scratch->set_memory (device_memory);
      filterbank->set_scratch (gpu_scratch);
    }
#endif

#define DUMP_FILTERBANK 0
#if DUMP_FILTERBANK
    Dump* dump = new Dump;
    dump->set_output( fopen("post_filterbank.dat", "w") );
    dump->set_input (convolved);
    operations.push_back (dump);
#endif

  }

  bool filterbank_after_dedisp
    = config->filterbank.get_convolve_when() == Filterbank::Config::Before;

  if (config->coherent_dedispersion &&
      config->filterbank.get_convolve_when() != Filterbank::Config::During)
  {
    if (!convolution)
      convolution = new Convolution;
    
    convolution->set_response (response);
    if (!config->single_pulse)
      convolution->set_passband (passband);
    
    if (filterbank_after_dedisp)
    {
      convolution->set_input  (unpacked);  
      convolution->set_output (unpacked);  // inplace
    }
    else
    {
      convolution->set_input  (convolved);  
      convolution->set_output (convolved);  // inplace
    }
    
    operations.push_back (convolution.get());
  }

  if (filterbank_after_dedisp)
    prepare_interchan (unpacked);
  else
    prepare_interchan (convolved);

  if (filterbank_after_dedisp && filterbank)
    operations.push_back (filterbank.get());

  if (config->plfb_nbin)
  {

    bool subints = config->single_pulse || config->integration_length;

    // Set up output
    Archiver* archiver = new Archiver;
    unloader.resize(1);
    unloader[0] = archiver;
    prepare_archiver( archiver );

    if (!phased_filterbank)
    {
      if (subints) 
      {

        Subint<PhaseLockedFilterbank> *sub_plfb = 
          new Subint<PhaseLockedFilterbank>;

        if (config->integration_length)
        {
          sub_plfb->set_subint_seconds (config->integration_length);
        }

        else if (config->single_pulse) 
        {
          sub_plfb->set_subint_turns (1);
          sub_plfb->set_fractional_pulses (config->fractional_pulses);
        }

        sub_plfb->set_unloader (unloader[0]);

        phased_filterbank = sub_plfb;

      }
      else
      {
        phased_filterbank = new PhaseLockedFilterbank;
      }
    }

    phased_filterbank->set_nbin (config->plfb_nbin);
    phased_filterbank->set_npol (config->npol);

    if (config->plfb_nchan)
      phased_filterbank->set_nchan (config->plfb_nchan);

    phased_filterbank->set_input (convolved);

    if (!phased_filterbank->has_output())
      phased_filterbank->set_output (new PhaseSeries);

    phased_filterbank->bin_divider.set_reference_phase(config->reference_phase);

    // Make dummy fold instance so that polycos get created
    fold.resize(1);
    fold[0] = new dsp::Fold;
    if (config->folding_period)
      fold[0]->set_folding_period (config->folding_period);
    if (config->ephemerides.size() > 0)
      fold[0]->set_pulsar_ephemeris ( config->ephemerides[0] );
    else if (config->predictors.size() > 0)
      fold[0]->set_folding_predictor ( config->predictors[0] );
    fold[0]->set_output ( phased_filterbank->get_output() );
    fold[0]->prepare ( manager->get_info() );

    operations.push_back (phased_filterbank.get());

    path.resize(1);
    path[0] = new SignalPath (operations);

    return;
    // the phase-locked filterbank does its own detection and folding
    
  }

  // peform zapping based on the results of the SKFilterbank
  if (config->sk_zap)
  { 

#if HAVE_CUDA
    if (run_on_gpu)
    {
      OperationThread::Wait * skthread_wait = skthread->get_wait();
      operations.push_back (skthread_wait);
    }
#endif

    SKMasker * skmasker = new SKMasker;
    if (!config->input_buffering)
      skmasker->set_buffering_policy (NULL);

#if HAVE_CUDA
    if (run_on_gpu)
    {
      // transfer the zap mask to the GPU
      BitSeries * skzapmask_on_gpu = new BitSeries();
      skzapmask_on_gpu->set_nbit (8);
      skzapmask_on_gpu->set_npol (1);
      skzapmask_on_gpu->set_nchan (config->filterbank.get_nchan());
      skzapmask_on_gpu->set_memory (device_memory);

      TransferBitSeriesCUDA* transfer = new TransferBitSeriesCUDA;
      transfer->set_kind( cudaMemcpyHostToDevice );
      transfer->set_input( skzapmask );
      transfer->set_output( skzapmask_on_gpu );
      operations.push_back (transfer);

      skmasker->set_mask_input (skzapmask_on_gpu);
      skmasker->set_engine (new CUDA::SKMaskerEngine (stream));
    }
    else
      skmasker->set_mask_input (skzapmask);
#else
    skmasker->set_mask_input (skzapmask);
#endif

    skmasker->set_input (convolved);
    skmasker->set_output (convolved);
    skmasker->set_M (config->sk_m);

    operations.push_back (skmasker);

  }

  // Cyclic spectrum also detects and folds
  if (config->cyclic_nchan) 
  {
    prepare_fold(convolved);
    return;
  }

  if (!detect)
    detect = new Detection;

  TimeSeries* detected = convolved;
  detect->set_input (convolved);
  detect->set_output (convolved);

#if HAVE_CUDA
  if (run_on_gpu)
  {
    config->ndim = 2;
    detect->set_engine (new CUDA::DetectionEngine(stream));
  }
#endif

  if (manager->get_info()->get_npol() == 1) 
  {
    cerr << "Only single polarization detection available" << endl;
    detect->set_output_state( Signal::PP_State );
  }
  else
  {
    if (config->fourth_moment)
    {
      detect->set_output_state (Signal::Stokes);
      detect->set_output_ndim (4);
    }
    else if (config->npol == 4)
    {
      detect->set_output_state (Signal::Coherence);
      detect->set_output_ndim (config->ndim);
    }
    else if (config->npol == 3)
      detect->set_output_state (Signal::NthPower);
    else if (config->npol == 2)
      detect->set_output_state (Signal::PPQQ);
    else if (config->npol == 1)
      detect->set_output_state (Signal::Intensity);
    else
      throw Error( InvalidState, "LoadToFold::construct",
                   "invalid npol config=%d input=%d",
                   config->npol, manager->get_info()->get_npol() );
  }

  operations.push_back (detect.get());

  if (config->npol == 3)
  {
    detected = new_time_series ();
    detect->set_output (detected);
  }
  else if (config->fourth_moment)
  {
    if (Operation::verbose)
      cerr << "LoadToFold::construct fourth order moments" << endl;
              
    FourthMoment* fourth = new FourthMoment;
    operations.push_back (fourth);

    fourth->set_input (detected);
    detected = new_time_series ();
    fourth->set_output (detected);
  }

  prepare_fold (detected);
}
catch (Error& error)
{
  throw error += "dsp::LoadToFold::construct";
}

void dsp::LoadToFold::prepare_interchan (TimeSeries* data)
{
  if (! config->interchan_dedispersion)
    return;

  if (Operation::verbose)
    cerr << "LoadToFold::prepare correct inter-channel dispersion delay"
         << endl;

  if (!sample_delay)
    sample_delay = new SampleDelay;

  sample_delay->set_input (data);
  sample_delay->set_output (data);
  sample_delay->set_function (new Dedispersion::SampleDelay);
  if (kernel)
    kernel->set_fractional_delay (true);

  operations.push_back (sample_delay.get());
}

double get_dispersion_measure (const Pulsar::Parameters* parameters)
{
  const Pulsar::TextParameters* teph;
  teph = dynamic_cast<const Pulsar::TextParameters*>(parameters);
  if (teph)
  {
    double dm = 0.0;
    teph->get_value (dm, "DM");
    return dm;
  }

  throw Error (InvalidState, "get_dispersion_measure (Pulsar::Parameters*)",
               "unknown Parameters class");
}

void dsp::LoadToFold::finalize ()
{
  assert (fold.size() > 0);

  const Pulsar::Predictor* predictor = 0;
  if (fold[0]->has_folding_predictor())
    predictor = fold[0]->get_folding_predictor();

  if (phased_filterbank)
    phased_filterbank->bin_divider.set_predictor( predictor );

  const Pulsar::Parameters* parameters = 0;
  if (fold[0]->has_pulsar_ephemeris())
    parameters = fold[0]->get_pulsar_ephemeris();

  double dm = 0.0;

  if (config->dispersion_measure)
  {
    dm = config->dispersion_measure;
    if (Operation::verbose)
      cerr << "LoadToFold::finalize config DM=" << dm << endl;
  }
  else if (parameters)
  {
    dm = get_dispersion_measure (parameters);
    if (Operation::verbose)
      cerr << "LoadToFold::finalize ephem DM=" << dm << endl;
  }

  if (config->coherent_dedispersion)
  {
    if (dm == 0.0)
      throw Error (InvalidState, "LoadToFold::finalize",
                   "coherent dedispersion enabled, but DM unknown");

    if (kernel)
      kernel->set_dispersion_measure (dm);
  }

  manager->get_info()->set_dispersion_measure( dm );

  // --repeat must reset the dm when the input is re-opened
  config->dispersion_measure = dm;

  /*
    In the case of unpacking two-bit data, set the corresponding
    parameters.  This is done in finalize because we really ought
    to set nsample to the largest number of samples smaller than the
    dispersion smearing, and in general the DM is known only after the
    ephemeris is prepared by Fold.
  */

  dsp::ExcisionUnpacker* excision;
  excision = dynamic_cast<dsp::ExcisionUnpacker*> ( manager->get_unpacker() );

  if (excision)
  {
    if ( config->excision_nsample )
      excision -> set_ndat_per_weight ( config->excision_nsample );
  
    if ( config->excision_threshold > 0 )
      excision -> set_threshold ( config->excision_threshold );
    
    if ( config->excision_cutoff >= 0 )
      excision -> set_cutoff_sigma ( config->excision_cutoff );
  }

  SingleThread::finalize ();

  for (unsigned ifold=0; ifold < fold.size(); ifold++)
  {
    Reference::To<Extensions> extensions = new Extensions;
    extensions->add_extension( path[ifold] );
    
    for (unsigned iop=0; iop < operations.size(); iop++)
      operations[iop]->add_extensions (extensions);
    
    fold[ifold]->get_output()->set_extensions (extensions);
  }

  // for now ...

  minimum_samples = 0;
  unsigned block_overlap = 0;

  bool report_vitals = thread_id==0 && config->report_vitals;

  if (kernel)
    cerr << "dspsr: dedispersion filter length=" << kernel->get_ndat ()
         << " (minimum=" << kernel->get_minimum_ndat () << ")" 
         << " complex samples" << endl;

  if (filterbank)
  {
    minimum_samples = filterbank->get_minimum_samples ();
    if (report_vitals)
    {
      cerr << "dspsr: " << config->filterbank.get_nchan() << " channel ";

      if (config->coherent_dedispersion &&
	  config->filterbank.get_convolve_when() == Filterbank::Config::During)
	cerr << "dedispersing ";
      else if (filterbank->get_freq_res() > 1)
        cerr << "by " << filterbank->get_freq_res() << " back ";

      cerr << "filterbank requires " << minimum_samples << " samples" << endl;
    }

    if (!config->input_buffering)
      block_overlap = filterbank->get_minimum_samples_lost ();
  }

  if (convolution)
  {
    minimum_samples = convolution->get_minimum_samples () * 
      convolution->get_input()->get_nchan();
    if (report_vitals)
      cerr << "dspsr: convolution requires at least " 
           << minimum_samples << " samples" << endl;

    if (!config->input_buffering)
      block_overlap = convolution->get_minimum_samples_lost () *
        convolution->get_input()->get_nchan();
  }

#if 0
  if (minimum_samples == 0)
    throw Error (InvalidState, "dsp::LoadToFold::finalize",
                 "minimum samples == 0");
#endif

  uint64_t block_size = ( minimum_samples - block_overlap )
    * config->get_times_minimum_ndat() + block_overlap;

  // set the block size to at least minimum_samples
  manager->set_maximum_RAM( config->get_maximum_RAM() );
  manager->set_minimum_RAM( config->get_minimum_RAM() );
  manager->set_copies( config->get_nbuffers() );

  if (block_overlap)
    manager->set_overlap( block_overlap );

  uint64_t ram = manager->set_block_size( block_size );

  // add the increased block size if the SKFB is being used
  if (skfilterbank)
  {
    int64_t skfb_increment = (int64_t) skfilterbank->get_skfb_inc (manager->get_input()->get_block_size());

    block_size += skfb_increment;
    block_overlap += skfb_increment;

    if (block_overlap)
      manager->set_overlap( block_overlap );
    ram = manager->set_block_size( block_size );

    skfb_increment *= -1;
    skresize->set_resize_samples (skfb_increment);

    if (Operation::verbose)
      cerr << "dsp::LoadToFold::finalize block_size will be adjusted by " 
          << skfb_increment << " samples for SKFB" << endl;
  }

  if (report_vitals)
  {
    double megabyte = 1024*1024;
    cerr << "dspsr: blocksize=" << manager->get_input()->get_block_size()
         << " samples or " << double(ram)/megabyte << " MB" << endl;
  }
}

void dsp::LoadToFold::end_of_data ()
{
  // ensure that remaining threads are not left waiting
  for (unsigned ifold=0; ifold < fold.size(); ifold++)
    fold[ifold]->finish();
}

void setup (const dsp::Fold* from, dsp::Fold* to)
{
  // copy over the output if there is one
  if (from && from->has_output())
    to->set_output( from->get_output() );

  if (from && from->has_folding_predictor())
    to->set_folding_predictor( from->get_folding_predictor() );

  if (from && from->has_pulsar_ephemeris())
    to->set_pulsar_ephemeris( from->get_pulsar_ephemeris() );

  if (!to->has_output())
    to->set_output( new dsp::PhaseSeries );
}

template<class T>
T* setup (dsp::Fold* ptr)
{
  // ensure that the current folder is of type T
  T* derived = dynamic_cast<T*> (ptr);

  if (!derived)
    derived = new T;

  setup (ptr, derived);

  return derived;
}

template<class T>
dsp::Fold* setup_not (dsp::Fold* ptr)
{
  // ensure that the current folder is not of type T
  T* derived = dynamic_cast<T*> (ptr);

  if (derived || !ptr)
    ptr = new dsp::Fold;

  setup (derived, ptr);

  return ptr;
}

const char* multifold_error =
  "Folding more than one pulsar and output archive filename set to\n"
  "\t%s\n"
  "The multiple output archives would over-write each other.\n";

void dsp::LoadToFold::prepare_fold (TimeSeries* to_fold)
{
  if (Operation::verbose)
    cerr << "dsp::LoadToFold::prepare_fold" << endl;

  if (config->pdmp_output)
  {
    Stats* stats = new Stats;
    stats->set_input (to_fold);
    operations.push_back (stats);
  }

  size_t nfold = 1 + config->additional_pulsars.size();

  nfold = std::max( nfold, config->predictors.size() );
  nfold = std::max( nfold, config->ephemerides.size() );

  if (nfold > 1 && !config->archive_filename.empty())
    throw Error (InvalidState, "dsp::LoadToFold::prepare_fold",
                 multifold_error, config->archive_filename.c_str());

  if (Operation::verbose)
    cerr << "dsp::LoadToFold::prepare_fold nfold=" << nfold << endl;

  fold.resize (nfold);
  path.resize (nfold);

  if (config->asynchronous_fold)
    asynch_fold.resize( nfold );

  bool subints = config->single_pulse || config->integration_length;

  if (manage_archiver)
  {
    if (subints)
      unloader.resize (nfold);
    else
      unloader.resize (1);
  }

  for (unsigned ifold=0; ifold < nfold; ifold++)
  {
    if (manage_archiver && ( ifold == 0 || subints ))
    {
      if (Operation::verbose)
        cerr << "dsp::LoadToFold::prepare_fold prepare Archiver" << endl;

      Archiver* archiver = new Archiver;
      unloader[ifold] = archiver;

      prepare_archiver( archiver );
    }

    if (subints)
    {
      if (Operation::verbose)
        cerr << "dsp::LoadToFold::prepare_fold prepare Subint" << endl;

      if (config->cyclic_nchan) 
      {

        Subint<CyclicFold>* subfold = 
          setup< Subint<CyclicFold> > (fold[ifold].ptr());

        subfold -> set_nchan(config->cyclic_nchan);
        subfold -> set_npol(config->npol);

        if (config->integration_length)
        {
          subfold -> set_subint_seconds (config->integration_length);

          if (config->minimum_integration_length > 0)
            unloader[ifold]->set_minimum_integration_length (config->minimum_integration_length);
        }
        else
          throw Error (InvalidState, "dsp::LoadToFold::prepare_fold", 
              "Single-pulse cyclic spectra not supported");

        subfold -> set_unloader (unloader[ifold]);

        fold[ifold] = subfold;

      }

      else 
      {
        Subint<Fold>* subfold = setup< Subint<Fold> > (fold[ifold].ptr());

        if (config->integration_length)
        {
          subfold -> set_subint_seconds (config->integration_length);

          if (config->minimum_integration_length > 0)
            unloader[ifold]->set_minimum_integration_length (config->minimum_integration_length);
        }
        else
        {
          subfold -> set_subint_turns (1);
          subfold -> set_fractional_pulses (config->fractional_pulses);
        }

        subfold -> set_unloader (unloader[ifold]);

        fold[ifold] = subfold;

      }

    }
    else
    {
      if (Operation::verbose)
        cerr << "dsp::LoadToFold::prepare_fold prepare Fold" << endl;

      fold[ifold] = setup_not< Subint<Fold> > (fold[ifold].ptr());
    }

    if (Operation::verbose)
      cerr << "dsp::LoadToFold::prepare_fold configuring" << endl;

    if (config->nbin)
    {
      fold[ifold]->set_nbin (config->nbin);
      fold[ifold]->set_force_sensible_nbin(config->force_sensible_nbin);
    }

    if (config->reference_phase)
      fold[ifold]->set_reference_phase (config->reference_phase);

    if (config->folding_period)
      fold[ifold]->set_folding_period (config->folding_period);

    Reference::To<ObservationChange> change;

    if (ifold && ifold <= config->additional_pulsars.size())
    {
      change = new ObservationChange;
      change->set_source( config->additional_pulsars[ifold-1] );
    }

    if (ifold < config->ephemerides.size())
    {
      if (!change)
        change = new ObservationChange;

      Pulsar::Parameters* ephem = config->ephemerides[ifold];
      change->set_source( ephem->get_name() );
      change->set_dispersion_measure( get_dispersion_measure(ephem) );

      fold[ifold]->set_pulsar_ephemeris ( config->ephemerides[ifold] );
    }

    if (ifold < config->predictors.size())
    {
      fold[ifold]->set_folding_predictor ( config->predictors[ifold] );

      Pulsar::SimplePredictor* simple
        = dynamic_kast<Pulsar::SimplePredictor>( config->predictors[ifold] );

      if (simple)
      {
        config->dispersion_measure = simple->get_dispersion_measure();

        if (!change)
          change = new ObservationChange;

        change->set_source( simple->get_name() );
        change->set_dispersion_measure( simple->get_dispersion_measure() );
      }
    }    

    fold[ifold]->set_input (to_fold);

    if (change)
      fold[ifold]->set_change (change);

    fold[ifold]->prepare ( manager->get_info() );

    if (ifold && ifold <= config->additional_pulsars.size())
    {
      if (!change)
        change = new ObservationChange;

      /*
        If additional pulsar names have been specified, then Fold::prepare
        will have retrieved an ephemeris, and the DM from this ephemeris 
        should make its way into the folded profile.
      */
      const Pulsar::Parameters* ephem = fold[ifold]->get_pulsar_ephemeris ();
      change->set_dispersion_measure( get_dispersion_measure(ephem) );
    }

    fold[ifold]->reset();

    path[ifold] = new SignalPath (operations);

    if (config->asynchronous_fold)
      asynch_fold[ifold] = new OperationThread (fold[ifold]);
    else
      operations.push_back( fold[ifold].get() );

#if HAVE_CUDA
    if (gpu_stream != undefined_stream)
    {
      cudaStream_t stream = (cudaStream_t) gpu_stream;
      fold[ifold]->set_engine (new CUDA::FoldEngine(stream, config->sk_zap));
    }
#endif

    path[ifold]->add( fold[ifold] );
  }

}

void dsp::LoadToFold::prepare_archiver( Archiver* archiver )
{
  bool subints = config->single_pulse || config->integration_length;
  bool multiple_outputs = subints 
    && ( (config->subints_per_archive>0) || (config->single_archive==false) );

  archiver->set_archive_class (config->archive_class.c_str());

  if (subints && config->single_archive)
  {
    cerr << "dspsr: Single archive with multiple sub-integrations" << endl;
    archiver->set_use_single_archive (true);
  }

  if (subints && config->subints_per_archive)
  {
    cerr << "dspsr: Archives with " << 
        config->subints_per_archive << " sub-integrations" << endl;
    archiver->set_use_single_archive (true);
    archiver->set_subints_per_file (config->subints_per_archive); 
  }

  if (config->single_pulse)
    archiver->set_store_dynamic_extensions (false);

  FilenameEpoch* epoch_convention = 0;
  FilenameSequential* index_convention = 0;

  if (config->single_pulse_archives())
    archiver->set_convention( new FilenamePulse );
  else
  {
    // If there is only one output file, use epoch convention.
    if (!multiple_outputs)
      archiver->set_convention( epoch_convention = new FilenameEpoch );

    // If archive_filename was specified, figure out whether
    // it represents a date string or not by looking for '%' 
    // characters.
    else if (!config->archive_filename.empty())
    {
      if (config->archive_filename.find('%') == string::npos)
        archiver->set_convention( index_convention = new FilenameSequential );
      else
        archiver->set_convention( epoch_convention = new FilenameEpoch );
    }

    // Default to epoch convention otherwise.
    else
      archiver->set_convention( epoch_convention = new FilenameEpoch );
  }

  if (epoch_convention && subints 
      && (config->single_archive || config->subints_per_archive))
    epoch_convention->report_unload = false;

  unsigned integer_seconds = unsigned(config->integration_length);

  if (config->integration_length && config->single_pulse)
    throw Error (InvalidState, "dsp::LoadToFold::prepare_archiver",
        "cannot set integration length in single pulse mode");

  if (config->integration_length &&
      config->integration_length == integer_seconds &&
      epoch_convention)
  {
    if (Operation::verbose)
      cerr << "dsp::LoadToFold::prepare_archiver integer_seconds="
           << integer_seconds << " in output filenames" << endl;

    epoch_convention->set_integer_seconds (integer_seconds);
  }

  if (!config->archive_filename.empty())
  {
    if (epoch_convention)
      epoch_convention->set_datestr_pattern (config->archive_filename);
    else if (index_convention)
      index_convention->set_base_filename (config->archive_filename);
    else
      throw Error (InvalidState, "dsp::LoadToFold::prepare_archiver",
                   "cannot set archive filename in single pulse mode");
  }

  archiver->set_archive_software( "dspsr" );

  if (sample_delay)
    archiver->set_archive_dedispersed (true);
  
  if (config->jobs.size())
    archiver->set_script (config->jobs);
  
  if (fold.size() > 1)
    archiver->set_path_add_source (true);
  
  if (!config->archive_extension.empty())
    archiver->set_extension (config->archive_extension);

  archiver->set_prefix( manager->get_input()->get_prefix() );
}

bool dsp::LoadToFold::output_subints () const
{
  return config && (config->single_pulse || config->integration_length);
}

void dsp::LoadToFold::share (SingleThread* other)
{
  SingleThread::share (other);

  LoadToFold* thread = dynamic_cast<LoadToFold*>( other );

  if (!thread)
    throw Error (InvalidParam, "dsp::LoadToFold::share",
		 "other thread is not a LoadToFold instance");

  //
  // clone the Fold/SubFold operations
  //
  // This code satisfies two preconditions:
  // 1) the folding operation may be either a Fold or SubFold class
  // 2) the folding operations should not share outputs or predictors
  //

  unsigned nfold = thread->fold.size();

  fold.resize( nfold );

  for (unsigned ifold = 0; ifold < nfold; ifold ++)
  {
    // the clone automatically copies the pointers to predictors ...
    fold[ifold] = thread->fold[ifold]->clone();
    
    // ... but each thread should have its own
    if (fold[ifold]->has_folding_predictor())
      fold[ifold]->set_folding_predictor
	(thread->fold[ifold]->get_folding_predictor()->clone());
    
    // ... and its own output.  New ones will be created in prepare()
    fold[ifold]->set_output( 0 );
    
    // and its own Fold::Engine
    fold[ifold]->set_engine( 0 ) ;
  }

  //
  // share the dedispersion kernel
  //
  kernel = thread->kernel;

  //
  // only the first thread must manage archival
  //
  if (output_subints())
    manage_archiver = false;
}

//! Run through the data
void dsp::LoadToFold::run ()
{
  if (log)
    for (unsigned iul=0; iul < unloader.size(); iul++)
      unloader[iul]->set_cerr (*log);
  
  SingleThread::run ();
}

//! Run through the data
void dsp::LoadToFold::finish () try
{
  if (phased_filterbank)
  {
    cerr << "Calling PhaseLockedFilterbank::normalize_output" << endl;
    phased_filterbank -> normalize_output ();
  }

  SingleThread::finish();

  bool subints = config->single_pulse || config->integration_length;

  if (!subints)
  {
    if (!unloader.size())
      throw Error (InvalidState, "dsp::LoadToFold::finish", "no unloader");

    for (unsigned i=0; i<fold.size(); i++)
    {
      Archiver* archiver = dynamic_cast<Archiver*>( unloader[0].get() );
      if (!archiver)
        throw Error (InvalidState, "dsp::LoadToFold::finish",
                     "unloader is not an archiver (single integration)");

      /*
        In multi-threaded applications, the thread that calls the
        finish method may not be the thread that called the prepare
        method.
      */

      if (Operation::verbose)
        cerr << "Creating archive " << i+1 << endl;

      if (phased_filterbank)
        archiver->unload( phased_filterbank->get_output() );
      else
        archiver->unload( fold[i]->get_result() );

    }
  }
}
catch (Error& error)
{
  throw error += "dsp::LoadToFold::finish";
}


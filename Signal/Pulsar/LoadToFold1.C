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
#include "dsp/OptimalFFT.h"

#if HAVE_CUDA
#include "dsp/FilterbankCUDA.h"
#include "dsp/OptimalFilterbank.h"
#include "dsp/TransferCUDA.h"
#include "dsp/DetectionCUDA.h"
#include "dsp/FoldCUDA.h"
#include "dsp/MemoryCUDA.h"
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
#include "pad.h"

#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

using namespace std;

static void* const undefined_stream = (void *) -1;

dsp::LoadToFold1::LoadToFold1 ()
  : cerr( std::cerr.rdbuf() ), error (InvalidState, "")
{
  manager = new IOManager;
  scratch = new Scratch;
  manage_archiver = true;
  log = 0;
  minimum_samples = 0;

  state = Idle;
  state_change = 0;
  thread_id = 0;
  share = 0;

  input_context = 0;
  gpu_stream = undefined_stream;
}

dsp::LoadToFold1::~LoadToFold1 ()
{
}

void dsp::LoadToFold1::take_ostream (std::ostream* newlog)
{
  if (newlog)
    this->cerr.rdbuf( newlog->rdbuf() );

  if (log)
    delete log;

  log = newlog;
}

//! Run through the data
void dsp::LoadToFold1::set_configuration (Config* configuration)
{
  config = configuration;
}

//! Set the Input from which data will be read
void dsp::LoadToFold1::set_input (Input* input)
{
  manager->set_input (input);
}

dsp::Input* dsp::LoadToFold1::get_input ()
{
  return manager->get_input ();
}

dsp::TimeSeries* dsp::LoadToFold1::new_time_series ()
{
  config->buffers ++;

  if (config->weighted_time_series)
  {
    if (Operation::verbose)
      cerr << "Creating WeightedTimeSeries instance" << endl;
    return new WeightedTimeSeries;
  }
  else
  {
    if (Operation::verbose)
      cerr << "Creating TimeSeries instance" << endl;
    return  new TimeSeries;
  }
}

void dsp::LoadToFold1::set_affinity (int core)
{
#if HAVE_SCHED_SETAFFINITY
  cpu_set_t set;
  CPU_ZERO (&set);
  CPU_SET (core, &set);

  pid_t tpid = syscall (SYS_gettid);

  //if (Operation::verbose)
    cerr << "dsp::LoadToFold1::set_affinity thread=" << thread_id
	 << " tpid=" << tpid << " core=" << core << endl;

  if (sched_setaffinity(tpid, sizeof(cpu_set_t), &set) < 0)
    throw Error (FailedSys, "dsp::LoadToFold1::set_affinity",
		 "sched_setaffinity (%d)", core);
#endif
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

void dsp::LoadToFold1::prepare () try
{
  TimeSeries::auto_delete = false;

  operations.resize (0);

  // each timeseries created will be counted in new_time_series
  config->buffers = 0;

  if (thread_id < config->affinity.size())
    set_affinity (config->affinity[thread_id]);

  // only the first thread should run input_prepare
  if (thread_id == 0 && config->input_prepare)
    config->input_prepare( manager->get_input() );

  if (!unpacked)
    unpacked = new_time_series();

  manager->set_output (unpacked);

  operations.push_back (manager.get());

#if HAVE_CUDA

  bool run_on_gpu = thread_id < config->get_cuda_ndevice();

  cudaStream_t stream = 0;

  if (run_on_gpu)
  {
    // disable input buffering when data must be copied between devices
    if (config->nthread > 1)
      config->input_buffering = false;

    int device = config->cuda_device[thread_id];
    cerr << "dspsr: thread " << thread_id 
	 << " using CUDA device " << device << endl;

    int ndevice = 0;
    cudaGetDeviceCount(&ndevice);

    if (device >= ndevice)
      throw Error (InvalidParam, "dsp::LoadToFold1::prepare",
		   "device=%d >= ndevice=%d", device, ndevice);

    cudaError err = cudaSetDevice (device);
    if (err != cudaSuccess)
      throw Error (InvalidState, "dsp::LoadToFold1::prepare",
		   "cudaMalloc failed: %s", cudaGetErrorString(err));

    unsigned nstream = count (config->cuda_device, (unsigned)device);

    if (nstream > 1)
    {
      cudaStreamCreate( &stream );
      cerr << "dspsr: thread " << thread_id << " on stream " << stream << endl;
    }

    gpu_stream = stream;

    device_memory = new CUDA::DeviceMemory (stream);

    Unpacker* unpacker = manager->get_unpacker ();
    if (unpacker->get_device_supported( device_memory ))
    {
      if (Operation::verbose)
	cerr << "LoadToFold1: unpack on GraphicsPU" << endl;

      unpacker->set_device( device_memory );
      unpacked->set_memory( device_memory );
	
      BitSeries* bits = new BitSeries;
      bits->set_memory (new CUDA::PinnedMemory);
      manager->set_output (bits);
    }
    else
    {
      if (Operation::verbose)
	cerr << "LoadToFold1: unpack on CPU" << endl;

      TransferCUDA* transfer = new TransferCUDA;
      transfer->set_kind( cudaMemcpyHostToDevice );
      transfer->set_input( unpacked );
	
      unpacked = new_time_series ();
      unpacked->set_memory (device_memory);
      transfer->set_output( unpacked );
      operations.push_back (transfer);
    }    
  }

#endif // HAVE_CUFFT

#define DUMP_UNPACKED 0
#if DUMP_UNPACKED
  Dump* dump = new Dump;
  dump->set_output( fopen("post_unpack.dat", "w") );
  dump->set_input(unpacked);
  operations.push_back (dump);
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
    prepare_final ();
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

    prepare_final();

    return;
    // the phase-locked filterbank does its own detection and folding
    
  }

  // Cyclic spectrum also detects and folds
  if (config->cyclic_nchan) 
  {
    prepare_fold(convolved);
    prepare_final();
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
      throw Error( InvalidState, "LoadToFold1::prepare",
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
      cerr << "LoadToFold1::prepare fourth order moments" << endl;
	      
    FourthMoment* fourth = new FourthMoment;
    operations.push_back (fourth);

    fourth->set_input (detected);
    detected = new_time_series ();
    fourth->set_output (detected);
  }

  prepare_fold (detected);
  prepare_final ();
}
catch (Error& error)
{
  throw error += "dsp::LoadToFold1::prepare";
}

void dsp::LoadToFold1::prepare_interchan (TimeSeries* data)
{
  if (! config->interchan_dedispersion)
    return;

  if (Operation::verbose)
    cerr << "LoadToFold1::prepare correct inter-channel dispersion delay"
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

void dsp::LoadToFold1::prepare_final ()
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
      cerr << "LoadToFold1::prepare_final config DM=" << dm << endl;
  }
  else if (parameters)
  {
    dm = get_dispersion_measure (parameters);
    if (Operation::verbose)
      cerr << "LoadToFold1::prepare_final ephem DM=" << dm << endl;
  }

  if (config->coherent_dedispersion)
  {
    if (dm == 0.0)
      throw Error (InvalidState, "LoadToFold1::prepare_final",
		   "coherent dedispersion enabled, but DM unknown");

    if (kernel)
      kernel->set_dispersion_measure (dm);
  }

  manager->get_info()->set_dispersion_measure( dm );

  // --repeat must reset the dm when the input is re-opened
  config->dispersion_measure = dm;

  /*
    In the case of unpacking two-bit data, set the corresponding
    parameters.  This is done in prepare_final because we really ought
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

  for (unsigned iop=0; iop < operations.size(); iop++)
    operations[iop]->prepare ();

  if (!config->single_pulse)
  for (unsigned ifold=0; ifold < fold.size(); ifold++)
  {
    Reference::To<Extensions> extensions = new Extensions;
    extensions->add_extension( path[ifold] );

    for (unsigned iop=0; iop < operations.size(); iop++)
      operations[iop]->add_extensions (extensions);

    fold[ifold]->get_output()->set_extensions (extensions);
  }

  for (unsigned idump=0; idump < config->dump_before.size(); idump++)
    insert_dump_point (config->dump_before[idump]);

  // for now ...

  minimum_samples = 0;
  unsigned block_overlap = 0;

  bool report_vitals = thread_id==0 && config->report_vitals;

  if (kernel && report_vitals)
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
    throw Error (InvalidState, "dsp::LoadToFold1::prepare_final",
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

  if (report_vitals)
  {
    double megabyte = 1024*1024;
    cerr << "dspsr: blocksize=" << manager->get_input()->get_block_size()
	 << " samples or " << double(ram)/megabyte << " MB" << endl;
  }

  for (unsigned iop=0; iop < operations.size(); iop++)
    operations[iop]->reserve ();
}

void dsp::LoadToFold1::insert_dump_point (const std::string& transform_name)
{
  typedef HasInput<TimeSeries> Xform;

  for (unsigned iop=0; iop < operations.size(); iop++)
  {
    if (operations[iop]->get_name() == transform_name)
    {
      Xform* xform = dynamic_cast<Xform*>( operations[iop].get() );
      if (!xform)
	throw Error (InvalidParam, "dsp::LoadToFold1::insert_dump_point",
		     transform_name + " does not have TimeSeries input");

      string filename = "pre_" + transform_name;

      if (config->nthread > 1)
	filename += "." + tostring (thread_id);

      filename += ".dump";

      cerr << "dspsr: dump output in " << filename << endl;

      Dump* dump = new Dump;
      dump->set_output( fopen(filename.c_str(), "w") );
      dump->set_input( xform->get_input() ) ;
      dump->set_output_binary (true);

      operations.push_back (dump);
      iop++;
    }
  }
}

uint64_t dsp::LoadToFold1::get_minimum_samples () const
{
  return minimum_samples;
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

void dsp::LoadToFold1::prepare_fold (TimeSeries* to_fold)
{
  if (Operation::verbose)
    cerr << "dsp::LoadToFold1::prepare_fold" << endl;

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
    throw Error (InvalidState, "dsp::LoadToFold1::prepare_fold",
		 multifold_error, config->archive_filename.c_str());

  if (Operation::verbose)
    cerr << "dsp::LoadToFold1::prepare_fold nfold=" << nfold << endl;

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
	cerr << "dsp::LoadToFold1::prepare_fold prepare Archiver" << endl;

      Archiver* archiver = new Archiver;
      unloader[ifold] = archiver;

      prepare_archiver( archiver );
    }

    if (subints)
    {
      if (Operation::verbose)
	cerr << "dsp::LoadToFold1::prepare_fold prepare Subint" << endl;

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
          throw Error (InvalidState, "dsp::LoadToFold1::prepare_fold", 
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
	cerr << "dsp::LoadToFold1::prepare_fold prepare Fold" << endl;

      fold[ifold] = setup_not< Subint<Fold> > (fold[ifold].ptr());
    }

    if (Operation::verbose)
      cerr << "dsp::LoadToFold1::prepare_fold configuring" << endl;

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
      fold[ifold]->set_engine (new CUDA::FoldEngine(stream));
    }
#endif

    path[ifold]->add( fold[ifold] );
  }

}

void dsp::LoadToFold1::prepare_archiver( Archiver* archiver )
{
  bool subints = config->single_pulse || config->integration_length;

  archiver->set_archive_class (config->archive_class.c_str());

  if (subints && config->single_archive)
  {
    cerr << "dspsr: Single archive with multiple sub-integrations" << endl;
    Pulsar::Archive* arch;
    arch = Pulsar::Archive::new_Archive (config->archive_class);
    archiver->set_archive (arch);
  }

  if (config->single_pulse)
    archiver->set_store_dynamic_extensions (false);

  FilenameEpoch* epoch_convention = 0;

  if (config->single_pulse_archives())
    archiver->set_convention( new FilenamePulse );
  else
    archiver->set_convention( epoch_convention = new FilenameEpoch );

  if (subints && config->single_archive)
    epoch_convention->report_unload = false;

  unsigned integer_seconds = unsigned(config->integration_length);

  if (config->integration_length &&
      config->integration_length == integer_seconds)
  {
    if (!epoch_convention)
      throw Error (InvalidState, "dsp::LoadToFold1::prepare_archiver",
		   "cannot set integration length in single pulse mode");

    if (Operation::verbose)
      cerr << "dsp::LoadToFold1::prepare_archiver integer_seconds="
           << integer_seconds << " in output filenames" << endl;

    epoch_convention->set_integer_seconds (integer_seconds);
  }

  if (!config->archive_filename.empty())
  {
    if (!epoch_convention)
      throw Error (InvalidState, "dsp::LoadToFold1::prepare_archiver",
		   "cannot set archive filename in single pulse mode");
    epoch_convention->set_datestr_pattern (config->archive_filename);
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

//! Run through the data
void dsp::LoadToFold1::run () try
{
  if (Operation::verbose)
    cerr << "dsp::LoadToFold1::run this=" << this 
         << " nops=" << operations.size() << endl;

  if (log)
  {
    scratch->set_cerr (*log);

    for (unsigned iul=0; iul < unloader.size(); iul++)
      unloader[iul]->set_cerr (*log);
  }

  // ensure that all operations are using the local log and scratch space
  for (unsigned iop=0; iop < operations.size(); iop++)
  {
    if (log)
    {
      cerr << "dsp::LoadToFold1::run " << operations[iop]->get_name() << endl;
      operations[iop] -> set_cerr (*log);
    }
    if (!operations[iop] -> scratch_was_set ())
      operations[iop] -> set_scratch (scratch);
  }

  Input* input = manager->get_input();

  uint64_t block_size = input->get_block_size();

  if (block_size == 0)
    throw Error (InvalidState, "dsp::LoadToFold1::run", "block_size=0");

  uint64_t total_samples = input->get_total_samples();
  uint64_t nblocks_tot = total_samples/block_size;

  unsigned block=0;

  int64_t last_decisecond = -1;

  bool finished = false;

while (!finished)
{
  while (!input->eod())
  {
    for (unsigned iop=0; iop < operations.size(); iop++) try
    {
      if (Operation::verbose)
	cerr << "dsp::LoadToFold1::run calling " 
	     << operations[iop]->get_name() << endl;
      
      operations[iop]->operate ();
      
      if (Operation::verbose)
	cerr << "dsp::LoadToFold1::run "
	     << operations[iop]->get_name() << " done" << endl;
      
    }
    catch (Error& error)
    {
      if (error.get_code() == EndOfFile)
        break;

      // ensure that remaining threads are not left waiting
      for (unsigned ifold=0; ifold < fold.size(); ifold++)
        fold[ifold]->finish();

      throw error += "dsp::LoadToFold1::run";
    }
    
    block++;
    
    if (thread_id==0 && config->report_done) 
    {
      double seconds = input->tell_seconds();
      int64_t decisecond = int64_t( seconds * 10 );
      
      if (decisecond > last_decisecond)
      {
        last_decisecond = decisecond;
        cerr << "Finished " << decisecond/10.0 << " s";

        if (nblocks_tot)
          cerr << " (" 
               << int (100.0*input->tell()/float(input->get_total_samples()))
               << "%)";

        cerr << "   \r";
      }
    }
  }

  finished = true;

  if (config->run_repeatedly)
  {
    ThreadContext::Lock context (input_context);

    if (config->repeated == 0 && input->tell() != 0)
    {
      // cerr << "dspsr: do it again" << endl;
      File* file = dynamic_cast<File*> (input);
      if (file)
      {
        finished = false;
        string filename = file->get_filename();
        file->close();
        // cerr << "file closed" << endl;
        file->open(filename);
        // cerr << "file opened" << endl;
        config->repeated = 1;

	if (config->input_prepare)
	  config->input_prepare (file);

	file->get_info()->set_dispersion_measure (config->dispersion_measure);
      }
    }
    else if (config->repeated)
    {
      config->repeated ++;
      finished = false;

      if (config->repeated == config->nthread)
        config->repeated = 0;
    }
  }
}

  if (Operation::verbose)
    cerr << "dsp::LoadToFold1::run end of data id=" << thread_id << endl;

  for (unsigned ifold=0; ifold < fold.size(); ifold++)
    fold[ifold]->finish();

  if (Operation::verbose)
    cerr << "dsp::LoadToFold1::run exit" << endl;
}
catch (Error& error)
{
  throw error += "dsp::LoadToFold1::run";
}

bool same_name (const dsp::Operation* A, const dsp::Operation* B)
{
  return A->get_name() == B->get_name();
}

template<typename C>
unsigned find_name (const C& container, unsigned i, const dsp::Operation* B)
{
  while (i < container.size() && ! same_name(container[i], B))
    i++;
  return i;
}

void dsp::LoadToFold1::combine (const LoadToFold1* that)
{
  if (Operation::verbose)
    cerr << "dsp::LoadToFold1::combine"
	 << " this size=" << operations.size() 
	 << " ptr=" << &(this->operations)
	 << " that size=" << that->operations.size()
	 << " ptr=" << &(that->operations) << endl;

  unsigned ithis = 0;
  unsigned ithat = 0;

  while (ithis < operations.size() && ithat < that->operations.size())
  {
    if (! same_name(operations[ithis], that->operations[ithat]))
    {
      // search for that in this
      unsigned jthis = find_name (operations, ithis, that->operations[ithat]);
      if (jthis == operations.size())
      {
	if (Operation::verbose)
	  cerr << "dsp::LoadToFold1::combine insert "
	       << that->operations[ithat]->get_name() << endl;

	// that was not found in this ... insert it and skip it
	operations.insert( operations.begin()+ithis, that->operations[ithat] );
	ithis ++;
	ithat ++;
      }
      else
      {
	// that was found later in this ... skip to it
	ithis = jthis;
      }

      continue;

#if 0
      if (operations[ithis]->get_function() != Operation::Procedural)
      {
	ithis ++;
	continue;
      }

      if (that->operations[ithat]->get_function() != Operation::Procedural)
      {
	ithat ++;
	continue;
      }

      throw Error (InvalidState, "dsp::LoadToFold1::combine",
		   "operation names do not match "
		   "'"+ operations[ithis]->get_name()+"'"
		   " != '"+that->operations[ithat]->get_name()+"'");
#endif
    }

    if (Operation::verbose)
      cerr << "dsp::LoadToFold1::combine "
	   << operations[ithis]->get_name() << endl;

    operations[ithis]->combine( that->operations[ithat] );

    ithis ++;
    ithat ++;
  }

  if (ithis != operations.size() || ithat != that->operations.size())
    throw Error (InvalidState, "dsp::LoadToFold1::combine",
		 "processes have different numbers of operations");
}

//! Run through the data
void dsp::LoadToFold1::finish () try
{
  if (phased_filterbank)
  {
    cerr << "Calling PhaseLockedFilterbank::normalize_output" << endl;
    phased_filterbank -> normalize_output ();
  }

  if (Operation::record_time)
    for (unsigned iop=0; iop < operations.size(); iop++)
      operations[iop]->report();

  bool subints = config->single_pulse || config->integration_length;

  if (!subints)
  {
    if (!unloader.size())
      throw Error (InvalidState, "dsp::LoadToFold1::finish", "no unloader");

    for (unsigned i=0; i<fold.size(); i++)
    {
      Archiver* archiver = dynamic_cast<Archiver*>( unloader[0].get() );
      if (!archiver)
	throw Error (InvalidState, "dsp::LoadToFold1::finish",
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
  throw error += "dsp::LoadToFold1::finish";
}


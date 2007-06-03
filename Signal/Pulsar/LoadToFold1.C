/***************************************************************************
 *
 *   Copyright (C) 2007 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/LoadToFold1.h"
#include "dsp/LoadToFoldConfig.h"

#include "dsp/IOManager.h"
#include "dsp/Scratch.h"
#include "dsp/SetBufferingPolicy.h"

#include "dsp/TwoBitCorrection.h"
#include "dsp/WeightedTimeSeries.h"

#include "dsp/ResponseProduct.h"
#include "dsp/DedispersionSampleDelay.h"
#include "dsp/RFIFilter.h"

#include "dsp/Filterbank.h"
#include "dsp/SampleDelay.h"
#include "dsp/PhaseLockedFilterbank.h"
#include "dsp/Detection.h"

#include "dsp/SubFold.h"
#include "dsp/PhaseSeries.h"

#include "dsp/Archiver.h"

#include "Pulsar/Archive.h"
#include "Pulsar/Parameters.h"
// #include "Pulsar/SimplePredictor.h"

#include "Error.h"

using namespace std;

dsp::LoadToFold1::LoadToFold1 ()
  : cerr( std::cerr.rdbuf() )
{
  manager = new IOManager;
  scratch = new Scratch;
  report = 1;
  log = 0;
  minimum_samples = 0;
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

dsp::TimeSeries* dsp::LoadToFold1::new_time_series ()
{
  config->buffers ++;

  if (config->weighted_time_series) {
    if (Operation::verbose)
      cerr << "Creating WeightedTimeSeries instance" << endl;
    return new dsp::WeightedTimeSeries;
  }
  else {
    if (Operation::verbose)
      cerr << "Creating TimeSeries instance" << endl;
    return new dsp::TimeSeries;
  }
}


void dsp::LoadToFold1::prepare ()
{
  SetBufferingPolicy::policy = SetBufferingPolicy::Input;
  Operation::preserve_data = true;
  Operation::record_time = true;
  TimeSeries::auto_delete = false;

  operations.resize (0);

  // each timeseries created will be counted in new_time_series
  config->buffers = 0;

  if (!unpacked)
    unpacked = new_time_series();

  manager->set_output (unpacked);

  operations.push_back (manager.get());

  if (manager->get_info()->get_detected()) {
    prepare_fold (unpacked);
    prepare_final ();
    return;
  }

  // the data are not detected, so set up phase coherent reduction path

  if (config->coherent_dedispersion) {

    if (!kernel)
      kernel = new Dedispersion;

    if (config->nfft)
      kernel->set_frequency_resolution (config->nfft);

    if (config->fres)
      kernel->set_frequency_resolution (config->fres);

  }
  else
    kernel = 0;

  if (!config->single_pulse && !passband)
    passband = new Response;

  Response* response = kernel.ptr();

  if (config->zap_rfi) {

    if (!rfi_filter)
      rfi_filter = new RFIFilter;

    rfi_filter->set_input (manager);

    response = rfi_filter;

    if (kernel) {

      if (!response_product)
	response_product = new ResponseProduct;

      response_product->add_response (kernel);
      response_product->add_response (rfi_filter);

      response = response_product;

    }


  }

  // only the Filterbank must be out-of-place
  TimeSeries* convolved = unpacked;

  if (config->nchan > 1) {

    // new storage for filterbank output (must be out-of-place)
    convolved = new_time_series ();

    // software filterbank constructor
    if (!filterbank)
      filterbank = new Filterbank;

    filterbank->set_input (unpacked);
    filterbank->set_output (convolved);
    filterbank->set_nchan (config->nchan);
    
    if (config->simultaneous_filterbank) {
      filterbank->set_response (response);
      filterbank->set_passband (passband);
    }
    
    operations.push_back (filterbank.get());
  }

  if (config->nchan == 1 || !config->simultaneous_filterbank) {
    
    if (!convolution)
      convolution = new Convolution;
    
    convolution->set_response (response);
    convolution->set_passband (passband);
    
    convolution->set_input  (convolved);  
    convolution->set_output (convolved);  // inplace
    
    operations.push_back (convolution.get());

  }

  if (config->interchan_dedispersion) {

    if (!sample_delay)
      sample_delay = new SampleDelay;

    sample_delay->set_input (convolved);
    sample_delay->set_output (convolved);
    sample_delay->set_function (new Dedispersion::SampleDelay);
    if (kernel)
      kernel->set_fractional_delay (true);

    operations.push_back (sample_delay.get());

  }

  if (config->plfb_nbin) {

    if (!phased_filterbank)
      phased_filterbank = new PhaseLockedFilterbank;

    phased_filterbank->set_nbin (config->plfb_nbin);

    if (config->plfb_nchan)
      phased_filterbank->set_nchan (config->plfb_nchan);

    phased_filterbank->set_input (convolved);

    if (!phased_filterbank->has_output())
      phased_filterbank->set_output (new PhaseSeries);

    phased_filterbank->divider.set_reference_phase (config->reference_phase);

    operations.push_back (phased_filterbank.get());

    // TO-DO: Archiver?

    return;

    // the phase-locked filterbank does its own detection and folding
  
  }
 
  TimeSeries* detected = NULL;
  
  if (!detect)
    detect = new Detection;
  
  if (config->npol == 4) {
    
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
    throw Error (InvalidState, "LoadToFold1::prepare",
		 "invalid config->npol=%d", config->npol);
  
  operations.push_back (detect.get());
  
  if (config->npol == 3) {
    TimeSeries* detected = new_time_series ();
    detect->set_input (convolved);
    detect->set_output (detected);
    prepare_fold (detected);
  }
  else {
    detect->set_input (convolved);
    detect->set_output (convolved);
    prepare_fold (convolved);
  }
  
  prepare_final ();
}

void dsp::LoadToFold1::prepare_final ()
{
  assert (fold.size() > 0);

  const Pulsar::Predictor* predictor = 0;
  if (fold[0]->has_folding_predictor())
    predictor = fold[0]->get_folding_predictor();

  if (phased_filterbank)
    phased_filterbank->divider.set_predictor( predictor );

  const Pulsar::Parameters* parameters = 0;
  if (fold[0]->has_pulsar_ephemeris())
    parameters = fold[0]->get_pulsar_ephemeris();

  double dm = 0.0;

  if (config->dispersion_measure) {
    dm = config->dispersion_measure;
    if (Operation::verbose)
      cerr << "LoadToFold1::prepare_final user DM=" << dm << endl;
  }

  else if (parameters) {
    dm = parameters->get_dispersion_measure ();
    if (Operation::verbose)
      cerr << "LoadToFold1::prepare_final eph DM=" << dm << endl;
  }

  if (config->coherent_dedispersion) {

    if (dm == 0.0)
      throw Error (InvalidState, "LoadToFold1::prepare_final",
		   "coherent dedispersion enabled, but DM unknown");

    if (kernel)
      kernel->set_dispersion_measure (dm);

  }

  /*
    In the case of unpacking two-bit data, set the corresponding
    parameters.  This is done in prepare_final because we really ought
    to set nsample to the largest number of samples smaller than the
    dispersion smearing, and in general the DM is known only after the
    ephemeris is prepared by Fold.
  */

  dsp::TwoBitCorrection* tbc;
  tbc = dynamic_cast<dsp::TwoBitCorrection*> ( manager->get_unpacker() );
    
  if ( tbc && config->tbc_nsample )
    tbc -> set_nsample ( config->tbc_nsample );
  
  if ( tbc && config->tbc_threshold )
    tbc -> set_threshold ( config->tbc_threshold );
  
  if ( tbc && config->tbc_cutoff )
    tbc -> set_cutoff_sigma ( config->tbc_cutoff );

  for (unsigned iop=0; iop < operations.size(); iop++) {

    Operation* op = operations[iop];
    op->prepare ();

  }

  // for now ...

  minimum_samples = 0;

  if (filterbank)
    minimum_samples = filterbank->get_minimum_samples ();

  if (convolution)
    minimum_samples = convolution->get_minimum_samples ();

  // assume that SampleDelay can handle any required buffering

}

uint64 dsp::LoadToFold1::get_minimum_samples () const
{
  return minimum_samples;
}

void setup_output (const dsp::Fold* from, dsp::Fold* to)
{
  // copy over the output if there is one
  if (from && from->has_output())
    to->set_output( from->get_output() );

  if (!to->has_output())
    to->set_output( new dsp::PhaseSeries );
}

template<class T>
T* setup (dsp::Fold* ptr)
{
  // ensure that the current folder is a single pulse folder
  T* derived = dynamic_cast<T*> (ptr);

  if (!derived)
    derived = new T;

  setup_output (ptr, derived);

  return derived;
}

template<class T>
dsp::Fold* setup_not (dsp::Fold* ptr)
{
  // ensure that the current folder is a single pulse folder
  T* derived = dynamic_cast<T*> (ptr);

  if (derived || !ptr)
    ptr = new dsp::Fold;

  setup_output (derived, ptr);

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

  size_t nfold = 1 + config->additional_pulsars.size();

  nfold = std::max( nfold, config->predictors.size() );
  nfold = std::max( nfold, config->ephemerides.size() );

  if (nfold > 1 && !config->archive_filename.empty())
    throw Error (InvalidState, "dsp::LoadToFold1::prepare_fold",
		 multifold_error, config->archive_filename.c_str());

  if (Operation::verbose)
    cerr << "dsp::LoadToFold1::prepare_fold nfold=1" << endl;

  fold.resize (nfold);

  if (config->single_pulse)
    archiver.resize (nfold);
  else
    archiver.resize (1);

  for (unsigned ifold=0; ifold < nfold; ifold++) {

    if (ifold == 0 || config->single_pulse) {

      if (Operation::verbose)
	cerr << "dsp::LoadToFold1::prepare_fold prepare Archiver" << endl;

      if (!archiver[ifold])
	archiver[ifold] = new Archiver;

      archiver[ifold]->set_archive_class (config->archive_class.c_str());
      
      if (!config->script.empty())
	archiver[ifold]->set_script (config->script);
      
      if (config->additional_pulsars.size())
	archiver[ifold]->set_source_filename (true);
      
      if (sample_delay)
	archiver[ifold]->set_archive_dedispersed (true);
      
      if (!config->archive_filename.empty())
	archiver[ifold]->set_filename (config->archive_filename);

    }

    if (config->single_pulse) {

      if (Operation::verbose)
	cerr << "dsp::LoadToFold1::prepare_fold prepare SubFold" << endl;

      SubFold* subfold = setup<SubFold> (fold[ifold].ptr());

      if (config->integration_length)
	subfold -> set_subint_seconds (config->integration_length);
      else
	subfold -> set_subint_turns (1);

      subfold -> set_unloader (archiver[ifold]);

      fold[ifold] = subfold;

      if (config->single_archive) {
	cerr << "Creating single pulse single archive" << endl;
	Pulsar::Archive* arch;
	arch = Pulsar::Archive::new_Archive (config->archive_class);
	archiver[ifold]->set_archive (arch);
      }

    }

    else {

      if (Operation::verbose)
	cerr << "dsp::LoadToFold1::prepare_fold prepare Fold" << endl;

      fold[ifold] = setup_not<SubFold> (fold[ifold].ptr());

    }

    if (Operation::verbose)
      cerr << "dsp::LoadToFold1::prepare_fold configuring" << endl;

    if (config->nbin)
      fold[ifold]->set_nbin (config->nbin);

    if (config->reference_phase)
      fold[ifold]->set_reference_phase (config->reference_phase);

    if (config->folding_period)
      fold[ifold]->set_folding_period (config->folding_period);

    if (ifold && ifold <= config->additional_pulsars.size())
      fold[ifold]->set_source_name ( config->additional_pulsars[ifold-1] );

    if (ifold < config->ephemerides.size())
      fold[ifold]->set_pulsar_ephemeris ( config->ephemerides[ifold] );

    if (ifold < config->predictors.size()) {

      fold[ifold]->set_folding_predictor ( config->predictors[ifold] );

#if 0
      Pulsar::SimplePredictor* simple_predictor
	= dynamic_kast<Pulsar::SimplePredictor>( config->predictors[ifold] );

      if (simple_predictor) {

	manager->get_info()->set_source
	  ( simple_predictor->get_name() );

	manager->get_info()->set_coordinates
	  ( simple_predictor->get_coordinates() );

	if (kernel) 
	  kernel->set_dispersion_measure
	    ( simple_predictor->get_dispersion_measure() );

      }
#endif

    }    

    fold[ifold]->set_input (to_fold);

    fold[ifold]->prepare ( manager->get_info() );

    fold[ifold]->get_output()->zero();

    operations.push_back( fold[ifold].get() );
  }

}

//! Run through the data
void dsp::LoadToFold1::run ()
{
  if (Operation::verbose)
    cerr << "dsp::LoadToFold1::run this=" << this 
	 << " nops=" << operations.size() << endl;

  // ensure that all operations are using the local log an scratch space
  for (unsigned iop=0; iop < operations.size(); iop++) {
    if (log) {
      cerr << "dsp::LoadToFold1::run " << operations[iop]->get_name() << endl;
      operations[iop] -> set_ostream (*log);
    }
    operations[iop] -> set_scratch (scratch);
  }

  Input* input = manager->get_input();

  uint64_t block_size = input->get_block_size();
  uint64_t total_samples = input->get_total_samples();
  uint64_t nblocks_tot = total_samples/block_size;

  unsigned block=0;

  int last_percent = -1;

  bool still_going = true;

  while (!input->eod() && still_going) {

    for (unsigned iop=0; iop < operations.size(); iop++) try {
      
      if (Operation::verbose)
	cerr << "dsp::LoadToFold1::run calling " 
	     << operations[iop]->get_name() << endl;
      
      operations[iop]->operate ();
      
      if (Operation::verbose)
	cerr << "dsp::LoadToFold1::run "
	     << operations[iop]->get_name() << " done" << endl;
      
    }
    catch (Error& error) {
      throw error += "dsp::LoadToFold1::run";
    }
    
    block++;
    
    if (report) {

      // report is set to the number of active threads
      int percent = int (100.0*report*float(block)/float(nblocks_tot));
      
      if (percent > last_percent) {
	cerr << "Finished " << percent << "%\r";
	last_percent = percent;
      }
      
    }
    
  }

  if (log)
    *log << "dsp::LoadToFold1::run end of data" << endl;

  if (Operation::verbose)
    cerr << "end of data" << endl;
  
  if (report) {
    fprintf (stderr, "%15s %15s %15s\n", "Operation","Time Spent","Discarded");
    for (unsigned iop=0; iop < operations.size(); iop++)
      fprintf (stderr, "%15s %15.2g %15d\n",
	       operations[iop]->get_name().c_str(),
	       (float) operations[iop]->get_total_time(),
	       (int) operations[iop]->get_discarded_weights()); 
  }
}

//! Run through the data
void dsp::LoadToFold1::finish ()
{
  if (phased_filterbank)  {
    cerr << "Calling PhaseLockedFilterbank::normalize_output" << endl;
    phased_filterbank -> normalize_output ();
  }

  if (!config->single_pulse) {

    for (unsigned i=0; i<fold.size(); i++) {

      if (Operation::verbose)
	cerr << "Creating archive " << i+1 << endl;

      archiver[0]->set_profiles (fold[i]->get_output());
      archiver[0]->set_archive_software( "dspsr" );
      archiver[0]->unload ();
      
    }

  }
  else if (config->single_archive) {

    for (unsigned i=0; i<archiver.size(); i++) {

      Pulsar::Archive* archive = archiver[i]->get_archive ();

      cerr << "Unloading single archive with " << archive->get_nsubint ()
	   << " integrations" << endl
	 << "Filename = '" << archive->get_filename() << "'" << endl;
    
      archive->unload ();
    
    }

  }

}

/***************************************************************************
 *
 *   Copyright (C) 2007 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/LoadToFold1.h"
#include "dsp/LoadToFoldConfig.h"

#include "dsp/IOManager.h"
#include "dsp/Input.h"
#include "dsp/Scratch.h"

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
#include "Pulsar/SimplePredictor.h"

#include "Error.h"
#include "pad.h"

using namespace std;

dsp::LoadToFold1::LoadToFold1 ()
  : cerr( std::cerr.rdbuf() ), error (InvalidState, "")
{
  manager = new IOManager;
  scratch = new Scratch;
  manage_archiver = true;
  log = 0;
  minimum_samples = 0;
  status = 0;
  id = 0;
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
    return new WeightedTimeSeries;
  }
  else {
    if (Operation::verbose)
      cerr << "Creating TimeSeries instance" << endl;
    return  new TimeSeries;
  }
}


void dsp::LoadToFold1::prepare () try
{
  // SetBufferingPolicy::policy = SetBufferingPolicy::Input;
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

  if (manager->get_info()->get_detected())
  {
    prepare_fold (unpacked);
    prepare_final ();
    return;
  }

  if (manager->get_info()->get_type() != Signal::Pulsar)
  {
    // the kernel gets messed up by DM=0 sources, like PolnCal
    if (id==0 && config->report)
      cerr << "Disabling coherent dedispersion of non-pulsar signal" << endl;
    config->coherent_dedispersion = false;
  }

  // the data are not detected, so set up phase coherent reduction path

  if (config->coherent_dedispersion)
  {
    if (!kernel)
      kernel = new Dedispersion;

    if (config->nfft)
      kernel->set_frequency_resolution (config->nfft);

    if (config->times_minimum_nfft)
      kernel->set_times_minimum_nfft (config->times_minimum_nfft);

    if (config->nsmear)
    {
      cerr << "setting smearing samples to " << config->nsmear << endl;
      kernel->set_smearing_samples (config->nsmear);
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

  // only the Filterbank must be out-of-place
  TimeSeries* convolved = unpacked;

  if (config->nchan > 1)
  {
    // new storage for filterbank output (must be out-of-place)
    convolved = new_time_series ();

    // software filterbank constructor
    if (!filterbank)
      filterbank = new Filterbank;

    filterbank->set_input (unpacked);
    filterbank->set_output (convolved);
    filterbank->set_nchan (config->nchan);
    
    if (config->simultaneous_filterbank)
    {
      filterbank->set_response (response);
      if (!config->single_pulse)
        filterbank->set_passband (passband);
    }

    if (config->nfft)
      filterbank->set_frequency_resolution (config->nfft);

    operations.push_back (filterbank.get());
  }

  if (config->coherent_dedispersion &&
      ( config->nchan == 1 || !config->simultaneous_filterbank ))
  {
    if (!convolution)
      convolution = new Convolution;
    
    convolution->set_response (response);
    if (!config->single_pulse)
      convolution->set_passband (passband);
    
    convolution->set_input  (convolved);  
    convolution->set_output (convolved);  // inplace
    
    operations.push_back (convolution.get());

  }

  if (config->interchan_dedispersion)
  {
    if (!sample_delay)
      sample_delay = new SampleDelay;

    sample_delay->set_input (convolved);
    sample_delay->set_output (convolved);
    sample_delay->set_function (new Dedispersion::SampleDelay);
    if (kernel)
      kernel->set_fractional_delay (true);

    operations.push_back (sample_delay.get());
  }

  if (config->plfb_nbin)
  {
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
 
  if (!detect)
    detect = new Detection;

  if (manager->get_info()->get_npol() == 1) 
  {
    cerr << "Only single polarization detection available" << endl;
    detect->set_output_state( Signal::PP_State );
  }
  else
  {
    if (config->npol == 4)
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
    TimeSeries* detected = new_time_series ();
    detect->set_input (convolved);
    detect->set_output (detected);
    prepare_fold (detected);
  }
  else
  {
    detect->set_input (convolved);
    detect->set_output (convolved);
    prepare_fold (convolved);
  }
  
  prepare_final ();
}
catch (Error& error)
{
  throw error += "dsp::LoadToFold1::prepare";
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

  if (config->dispersion_measure)
  {
    dm = config->dispersion_measure;
    if (Operation::verbose)
      cerr << "LoadToFold1::prepare_final user DM=" << dm << endl;
  }

  else if (parameters)
  {
    dm = parameters->get_dispersion_measure ();
    if (Operation::verbose)
      cerr << "LoadToFold1::prepare_final eph DM=" << dm << endl;
  }

  if (config->coherent_dedispersion)
  {
    if (dm == 0.0)
      throw Error (InvalidState, "LoadToFold1::prepare_final",
		   "coherent dedispersion enabled, but DM unknown");

    if (kernel)
      kernel->set_dispersion_measure (dm);
  }

  manager->get_info()->set_between_channel_dm( dm );
  manager->get_info()->set_dispersion_measure( dm );

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
    tbc -> set_ndat_per_weight ( config->tbc_nsample );
  
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

  if (filterbank) {
    // cerr << "ASK FILTERBANK" << endl;
    minimum_samples = filterbank->get_minimum_samples ();
  }

  if (convolution) {
    // cerr << "ASK CONVOLUTION" << endl;
    minimum_samples = convolution->get_minimum_samples ();
  }

  // cerr << "MINIMUM SAMPLES=" << minimum_samples << endl;

  // set the block size to at least minimum_samples
  uint64 ram = manager->set_block_size
    ( minimum_samples * config->get_times_minimum_ndat(),
      config->get_maximum_RAM(),
      config->get_nbuffers() );

  if (id==0 && config->report)
  {
    double megabyte = 1024*1024;
    cerr << "dspsr: blocksize=" << manager->get_input()->get_block_size()
	 << " samples or " << double(ram)/megabyte << " MB" << endl;
  }
}

uint64 dsp::LoadToFold1::get_minimum_samples () const
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

  size_t nfold = 1 + config->additional_pulsars.size();

  nfold = std::max( nfold, config->predictors.size() );
  nfold = std::max( nfold, config->ephemerides.size() );

  if (nfold > 1 && !config->archive_filename.empty())
    throw Error (InvalidState, "dsp::LoadToFold1::prepare_fold",
		 multifold_error, config->archive_filename.c_str());

  if (Operation::verbose)
    cerr << "dsp::LoadToFold1::prepare_fold nfold=" << nfold << endl;

  fold.resize (nfold);

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
	cerr << "dsp::LoadToFold1::prepare_fold prepare SubFold" << endl;

      SubFold* subfold = setup<SubFold> (fold[ifold].ptr());

      if (config->integration_length)
	subfold -> set_subint_seconds (config->integration_length);
      else
      {
	subfold -> set_subint_turns (1);
	subfold -> set_fractional_pulses (config->fractional_pulses);
      }

      subfold -> set_unloader (unloader[ifold]);

      fold[ifold] = subfold;

    }
    else
    {
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

    if (ifold < config->predictors.size())
    {
      fold[ifold]->set_folding_predictor ( config->predictors[ifold] );

      Pulsar::SimplePredictor* simple
	= dynamic_kast<Pulsar::SimplePredictor>( config->predictors[ifold] );

      if (simple)
      {
	manager->get_info()->set_source
	  ( simple->get_name() );

	manager->get_info()->set_coordinates
	  ( simple->get_coordinates() );

	config->dispersion_measure = simple->get_dispersion_measure();
      }
    }    

    fold[ifold]->set_input (to_fold);

    fold[ifold]->prepare ( manager->get_info() );

    fold[ifold]->get_output()->zero();

    operations.push_back( fold[ifold].get() );
  }

}

void dsp::LoadToFold1::prepare_archiver( Archiver* archiver )
{
  bool subints = config->single_pulse || config->integration_length;

  archiver->set_archive_class (config->archive_class.c_str());

  if (subints && config->single_archive)
  {
    cerr << "Single archive with multiple sub-integrations" << endl;
    Pulsar::Archive* arch;
    arch = Pulsar::Archive::new_Archive (config->archive_class);
    archiver->set_archive (arch);
  }

  FilenameEpoch* epoch_convention = 0;

  if (config->single_pulse)
    archiver->set_convention( new FilenamePulse );
  else
    archiver->set_convention( epoch_convention = new FilenameEpoch );

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

  if (sample_delay)
    archiver->set_archive_dedispersed (true);
  
  if (!config->script.empty())
    archiver->set_script (config->script);
  
  if (config->additional_pulsars.size())
    archiver->set_path_add_source (true);
  
  if (!config->archive_extension.empty())
    archiver->set_extension (config->archive_extension);
}

//! Run through the data
void dsp::LoadToFold1::run () try
{
  if (Operation::verbose)
    cerr << "dsp::LoadToFold1::run this=" << this 
	 << " nops=" << operations.size() << endl;

  if (log)
    scratch->set_ostream (*log);

  // ensure that all operations are using the local log and scratch space
  for (unsigned iop=0; iop < operations.size(); iop++)
  {
    if (log)
    {
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

  int64 last_decisecond = -1;

  bool still_going = true;

  while (!input->eod() && still_going)
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
      throw error += "dsp::LoadToFold1::run";
    }
    
    block++;
    
    if (id==0 && config->report) 
    {
      double seconds = input->tell_seconds();
      int64 decisecond = int64( seconds * 10 );
      
      if (decisecond > last_decisecond)
      {
	last_decisecond = decisecond;
	cerr << "Finished " << decisecond/10.0 << " s";

	if (nblocks_tot)
	  cerr << " (" 
	       << int (100.0*config->report*float(block)/float(nblocks_tot))
	       << "%)";

	cerr << "   \r";
      }
    }
  }

  if (Operation::verbose)
    cerr << "dsp::LoadToFold1::run end of data" << endl;

  for (unsigned ifold=0; ifold < fold.size(); ifold++)
    fold[ifold]->finish();

  unsigned cwidth = 20;

  if (Operation::verbose || (id==0 && config->report))
  {
    cerr << pad (cwidth, "Operation")
	 << pad (cwidth, "Time Spent")
	 << pad (cwidth, "Discarded") << endl;
    
    for (unsigned iop=0; iop < operations.size(); iop++)
      cerr << pad (cwidth, operations[iop]->get_name())
	   << pad (cwidth, tostring(operations[iop]->get_total_time()))
	   << pad (cwidth, tostring(operations[iop]->get_discarded_weights()))
	   << endl;
  }

  if (Operation::verbose)
    cerr << "dsp::LoadToFold1::run exit" << endl;
}
catch (Error& error)
{
  throw error += "dsp::LoadToFold1::run";
}

//! Run through the data
void dsp::LoadToFold1::finish () try
{
  if (phased_filterbank)
  {
    cerr << "Calling PhaseLockedFilterbank::normalize_output" << endl;
    phased_filterbank -> normalize_output ();
  }

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

      if (Operation::verbose)
	cerr << "Creating archive " << i+1 << endl;

      archiver->set_archive_software( "dspsr" );
      archiver->unload( fold[i]->get_output() );
    }
  }
  else if (config->single_archive)
  {
    for (unsigned i=0; i<unloader.size(); i++)
    {
      Archiver* archiver = dynamic_cast<Archiver*>( unloader[i].get() );
      if (!archiver)
	throw Error (InvalidState, "dsp::LoadToFold1::finish",
		     "unloader is not an archiver (single archive)");

      Pulsar::Archive* archive = archiver->get_archive ();

      cerr << "Unloading single archive with " << archive->get_nsubint ()
	   << " integrations" << endl
	 << "Filename = '" << archive->get_filename() << "'" << endl;
    
      archive->unload ();
    }
  }
}
catch (Error& error)
{
  throw error += "dsp::LoadToFold1::finish";
}


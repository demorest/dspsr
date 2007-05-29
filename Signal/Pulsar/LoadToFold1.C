#include "dsp/LoadToFold1.h"

#include "dsp/IOManager.h"
#include "dsp/MultiFile.h"
#include "dsp/SetBufferingPolicy.h"

#include "dsp/Unpacker.h"
#include "dsp/BitSeries.h"
#include "dsp/TwoBitCorrection.h"
#include "dsp/WeightedTimeSeries.h"

#include "dsp/ResponseProduct.h"
#include "dsp/Dedispersion.h"
#include "dsp/RFIFilter.h"

#include "dsp/AutoCorrelation.h"
#include "dsp/Filterbank.h"
#include "dsp/ACFilterbank.h"

#include "dsp/SampleDelay.h"

#include "dsp/PhaseLockedFilterbank.h"
#include "dsp/Detection.h"

#include "dsp/SubFold.h"
#include "dsp/PhaseSeries.h"

#include "dsp/Archiver.h"

#include "Pulsar/Archive.h"

#include "Error.h"

using namespace std;

dsp::LoadToFold1::LoadToFold1 ()
{
  manager = new dsp::IOManager;

  // these attributes may be copied to a (nested) Configuration class
  weighted_time_series = true;
}

dsp::LoadToFold1::~LoadToFold1 ()
{
}

//! Set the Input from which data will be read
void dsp::LoadToFold1::set_input (Input* input)
{
  manager->set_input (input);
}


dsp::TimeSeries* dsp::LoadToFold1::new_time_series ()
{
  if (weighted_time_series) {
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

//! Run through the data
void dsp::LoadToFold1::run ()
{
}

void dsp::LoadToFold1::prepare ()
{
  SetBufferingPolicy::policy = SetBufferingPolicy::Input;
  Operation::preserve_data = true;

  operations.resize (0);

  if (!unpacked)
    unpacked = new_time_series();

  manager->set_output (unpacked);

  operations.push_back (manager.get());

  if (manager->get_info()->get_detected()) {
    prepare_fold (unpacked);
    return;
  }

  // the data are not detected, so set up phase coherent reduction path

  if (coherent_dedispersion) {

    if (!kernel)
      kernel = new Dedispersion;

    if (nfft)
      kernel->set_frequency_resolution (nfft);

    if (fres)
      kernel->set_frequency_resolution (fres);

  }
  else
    kernel = 0;

  if (!single_pulse && !passband)
    passband = new Response;

  Response* response = kernel.ptr();

  if (rfi_filter) {

    ResponseProduct* product = new ResponseProduct;
    product->add_response (kernel);
    product->add_response (rfi_filter);
    response = product;

    rfi_filter->set_input (manager);

  }

  // only the Filterbank must be out-of-place
  TimeSeries* convolved = unpacked;

  if (nchan > 1) {

    // new storage for filterbank output (must be out-of-place)
    convolved = new_time_series ();

    // software filterbank constructor
    if (!filterbank)
      filterbank = new Filterbank;

    filterbank->set_input (unpacked);
    filterbank->set_output (convolved);
    filterbank->set_nchan (nchan);
    
    if (simultaneous_filterbank) {
      filterbank->set_response (response);
      filterbank->set_passband (passband);
    }
    
    operations.push_back (filterbank.get());
  }

  if (nchan == 1 || !simultaneous_filterbank) {
    
    if (!convolution)
      convolution = new Convolution;
    
    convolution->set_response (response);
    convolution->set_passband (passband);
    
    convolution->set_input  (convolved);  
    convolution->set_output (convolved);  // inplace
    
    operations.push_back (convolution.get());

  }

  if (sample_delay) {

    sample_delay->set_input (convolved);
    sample_delay->set_output (convolved);
    sample_delay->set_function (new Dedispersion::SampleDelay);
    if (kernel)
      kernel->set_fractional_delay (true);

    operations.push_back (sample_delay.get());

  }

  if (plfb_nbin) {

    if (!phased_filterbank)
      phased_filterbank = new PhaseLockedFilterbank;

    phased_filterbank->set_nbin (plfb_nbin);

    if (plfb_nchan)
      phased_filterbank->set_nchan (plfb_nchan);

    phased_filterbank->set_input (convolved);

    if (!phased_filterbank->has_output())
      phased_filterbank->set_output (new PhaseSeries);

    phased_filterbank->divider.set_reference_phase (reference_phase);

    operations.push_back (phased_filterbank.get());

    // TO-DO: Archiver?

    return;

    // the phase-locked filterbank does its own detection and folding
  
  }
 
  TimeSeries* detected = NULL;
  
  if (!detect)
    detect = new Detection;
  
  if (npol == 4) {
    
    detect->set_output_state (Signal::Coherence);
    detect->set_output_ndim (ndim);
    
  }
  else if (npol == 3)
    detect->set_output_state (Signal::NthPower);
  else if (npol == 2)
    detect->set_output_state (Signal::PPQQ);
  else if (npol == 1)
    detect->set_output_state (Signal::Intensity);
  else
    throw Error (InvalidState, "LoadToFold1::prepare",
		 "invalid npol=%d", npol);
  
  operations.push_back (detect.get());
  
  if (npol == 3) {
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
  
}

void setup_input (const dsp::Fold* from, dsp::Fold* to)
{
  // copy over the input if there is one
  if (from && from->has_output())
    to->set_input( from->get_output() );

  if (!to->has_input())
    to->set_input( new dsp::PhaseSeries );
}

template<class T>
T* setup (dsp::Fold* ptr)
{
  // ensure that the current folder is a single pulse folder
  T* derived = dynamic_cast<T*> (ptr);

  if (!derived)
    derived = new T;

  setup_input (ptr, derived);

  return derived;
}

template<class T>
dsp::Fold* setup_not (dsp::Fold* ptr)
{
  // ensure that the current folder is a single pulse folder
  T* derived = dynamic_cast<T*> (ptr);

  if (derived)
    ptr = new dsp::Fold;

  setup_input (derived, ptr);

  return ptr;
}

void dsp::LoadToFold1::prepare_fold (TimeSeries* to_fold)
{
  unsigned nfold = 1 + additional_pulsars.size();

  fold.resize (nfold);
  archiver.resize (nfold);

  for (unsigned ifold=0; ifold < nfold; ifold++) {

    if (!archiver[ifold])
      archiver[ifold] = new Archiver;

    archiver[ifold]->set_archive_class (archive_class.c_str());

    if (!script.empty())
      archiver[ifold]->set_script (script);

    if (additional_pulsars.size())
      archiver[ifold]->set_source_filename (true);

    if (single_pulse) {

      SubFold* subfold = setup<SubFold> (fold[ifold].ptr());

      if (integration_length)
	subfold -> set_subint_seconds (integration_length);
      else
	subfold -> set_subint_turns (1);

      subfold -> set_unloader (archiver[ifold]);

      fold[ifold] = subfold;

      if (single_archive) {
	cerr << "Creating single pulse single archive" << endl;
	Pulsar::Archive* arch = Pulsar::Archive::new_Archive (archive_class);
	archiver[ifold]->set_archive (arch);
      }

    }

    else
      fold[ifold] = setup_not<SubFold> (fold[ifold].ptr());

    if (nbin)
      fold[ifold]->set_nbin (nbin);

    if (reference_phase)
      fold[ifold]->set_reference_phase (reference_phase);

    if (folding_period)
      fold[ifold]->set_folding_period (folding_period);

    /*
      for (unsigned ieph=0; ieph < ephemerides.size(); ieph++)
      fold[ifold]->add_pulsar_ephemeris ( ephemerides[ieph] );
    */

    for (unsigned ipoly=0; ipoly < predictors.size(); ipoly++)
      fold[ifold]->add_folding_predictor ( predictors[ipoly] );
    
    fold[ifold]->set_input (to_fold);

  }

}



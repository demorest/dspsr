/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/ExcisionUnpacker.h"
#include "dsp/WeightedTimeSeries.h"
#include "dsp/Input.h"
#include "dsp/BitTable.h"

#include "templates.h"
#include "Error.h"
#include <assert.h>

using namespace std;

//! Null constructor
dsp::ExcisionUnpacker::ExcisionUnpacker (const char* _name)
  : HistUnpacker (_name)
{
  ndat_per_weight = 0;

  if (psrdisp_compatible)
  {
    cerr << "dsp::TwoBitCorrection psrdisp compatibility\n"
      "   using cutoff sigma of 6.0 instead of 10.0" << endl;
    cutoff_sigma = 6.0;
  }
  else
    cutoff_sigma = 10.0;

  // These are set in set_limits()
  nlow_min = 0;
  nlow_max = 0;

  built = false;  
}

void dsp::ExcisionUnpacker::set_output (TimeSeries* _output)
{
  if (verbose)
    cerr << "dsp::ExcisionUnpacker::set_output (" << _output << ")" << endl;

  Unpacker::set_output (_output);
  weighted_output = dynamic_cast<WeightedTimeSeries*> (_output);
}

//! Initialize and resize the output before calling unpack
void dsp::ExcisionUnpacker::reserve ()
{
  if (weighted_output)
  {
    weighted_output -> set_ndat_per_weight (get_ndat_per_weight());
    weighted_output -> set_nchan_weight (1);
    weighted_output -> set_npol_weight (input->get_npol());
  }

  output->resize ( input->get_ndat() );

  if (weighted_output)
    weighted_output -> neutral_weights ();
}

//! Match the unpacker to the resolution
void dsp::ExcisionUnpacker::match_resolution (const Input* input)
{
  unsigned resolution = input->get_resolution();
  if (resolution > ndat_per_weight)
    set_ndat_per_weight (resolution);
  else
    set_ndat_per_weight (multiple_greater (ndat_per_weight, resolution));
}

//! Return ndat_per_weight
unsigned dsp::ExcisionUnpacker::get_resolution () const
{
  return ndat_per_weight;
}


//! Set the number of time samples used to estimate digitized power
void dsp::ExcisionUnpacker::set_ndat_per_weight (unsigned _ndat)
{
  if (verbose)
    cerr << "dsp::ExcisionUnpacker::set_ndat_per_weight=" << _ndat << endl;

  // ndat_per_weight must equal nstate
  HistUnpacker::set_nstate (_ndat);
  ndat_per_weight = _ndat;
  built = false;
}

void dsp::ExcisionUnpacker::set_limits ()
{
  if (cutoff_sigma == 0.0)
  {
    nlow_min = 0;
    nlow_max = get_ndat_per_weight();
    return;
  }

  float fsample = get_ndat_per_weight();

  if (verbose)
    cerr << "dsp::ExcisionUnpacker::set_limits nsample=" << fsample << endl;

  float nlo_mean = fsample * ja98.get_mean_Phi ();
  float nlo_variance = fsample * ja98.get_var_Phi ();

  if (nlo_mean == fsample)
    throw Error (InvalidState, "dsp::ExcisionUnpacker::set_limits",
                 "sampling threshold error: mean nlow=%f == sample size=%f",
                 nlo_mean, fsample);

  // the root mean square deviation
  float nlo_sigma = sqrt( nlo_variance );

  if (verbose)
    cerr << "dsp::ExcisionUnpacker::set_limits nlo_mean=" << nlo_mean 
         << " nlo_sigma=" << nlo_sigma << endl;

  // backward compatibility
  if (psrdisp_compatible)
  {
    // in psrdisp, sigma was incorrectly set as
    nlo_sigma = sqrt( float(get_ndat_per_weight()) );

    cerr << "dsp::ExcisionUnpacker psrdisp compatibility\n"
      "   setting nlo_sigma to " << nlo_sigma << endl;
  }

  nlow_max = unsigned (nlo_mean + (cutoff_sigma * nlo_sigma));

  if (nlow_max >= get_ndat_per_weight())
  {
    if (verbose)
      cerr << "dsp::ExcisionUnpacker::set_limits resetting nmax:"
	   << nlow_max << " to ndat_per_weight-2:" << get_ndat_per_weight()-1
	   << endl;
    nlow_max = get_ndat_per_weight()-1;
  }

  if (cutoff_sigma * nlo_sigma >= nlo_mean+1.0)
  {
    if (verbose)
      cerr << "dsp::ExcisionUnpacker::set_limits resetting nmin:"
	   << nlow_min << " to one:1" << endl;
    nlow_min = 1;
  }
  else 
    nlow_min = unsigned (nlo_mean - (cutoff_sigma * nlo_sigma));
  
  if (verbose)
    cerr << "dsp::ExcisionUnpacker::set_limits nmin:"
         << nlow_min << " and nmax:" << nlow_max << endl;
}

void dsp::ExcisionUnpacker::build ()
{
  if (verbose)
    cerr << "dsp::ExcisionUnpacker::build" << endl;
  set_limits ();
  zero_histogram ();
  built = true;
}

void dsp::ExcisionUnpacker::not_built ()
{
  built = false;
}

void dsp::ExcisionUnpacker::unpack ()
{
  if (verbose)
    cerr << "dsp::ExcisionUnpacker::unpack" << endl;;

  uint64_t ndat = input->get_ndat();

  if (ndat < get_ndat_per_weight())
    return;

  // build the two-bit lookup table
  if (!built)
  {
    if (verbose)
      cerr << "dsp::Excision::unpack calling build" << endl;
    build ();
  }

  const unsigned char* rawptr = input->get_rawptr();

  unsigned ndig = get_ndig ();

  // weights are used only if output is a WeightedTimeseries
  unsigned* weights = 0;
  uint64_t nweights = 0;

  // the number of floating point numbers to unpack from each digitizer
  uint64_t nfloat = ndat * get_ndim_per_digitizer();

  for (unsigned idig=0; idig<ndig; idig++)
  {
    unsigned ipol = get_output_ipol (idig);
    unsigned ichan = get_output_ichan (idig);
    unsigned input_offset = get_input_offset (idig);
    unsigned output_offset = get_output_offset (idig);

#ifdef _DEBUG
    cerr << "idig=" << idig << " ichan=" << ichan << " ipol=" << ipol 
	 << "\n  offsets: input=" << input_offset 
         << " output=" << output_offset << endl;
#endif

    const unsigned char* from = rawptr + input_offset;

    float* into = output->get_datptr (ichan, ipol) + output_offset;

#ifdef _DEBUG
    cerr << "dsp::ExcisionUnpacker::unpack idig=" << idig << "/" << ndig
	 << " from=" << (void*)from << " to=" << into << endl;
#endif

    // if the output TimeSeries is a weighted output, use its weights array
    if (weighted_output)
    {
      weights = weighted_output -> get_weights (0, ipol);
      nweights = weighted_output -> get_nweights ();
    }

    unsigned long* hist = 0;
    if (keep_histogram)
      hist = get_histogram (idig, get_ndat_per_weight());

    current_digitizer = idig;

    dig_unpack (from, into, nfloat, hist, weights, nweights);
      
  }  // for each stream of digitized data


  if (weighted_output)
  {
    weighted_output -> mask_weights ();
    uint64_t nbad = weighted_output -> get_nzero ();
    discarded_weights += nbad;

    if (nbad && verbose)
      cerr << "dsp::ExcisionUnpacker::unpack " << nbad 
           << "/" << weighted_output -> get_nweights()
           << " total bad weights" << endl;

  }
}

/*! By default, the data from each polarization is interleaved byte by byte */
unsigned dsp::ExcisionUnpacker::get_input_offset (unsigned idig) const
{
  return idig;
}

/*! By default, the data from each polarization is interleaved byte by byte */
unsigned dsp::ExcisionUnpacker::get_input_incr () const 
{
  return input->get_npol() * get_output_incr();
}

/*! By default, the output from each digitizer is contiguous */
unsigned dsp::ExcisionUnpacker::get_output_incr () const
{
  if (get_ndim_per_digitizer () == 2)
    return 1;
  else
    return input->get_ndim();
}

//! Set the cut off power for impulsive interference excision
void dsp::ExcisionUnpacker::set_cutoff_sigma (float _cutoff_sigma)
{
  if (cutoff_sigma == _cutoff_sigma)
    return;

  if (verbose)
    cerr << "dsp::ExcisionUnpacker::set_cutoff_sigma = "<<_cutoff_sigma<<endl;

  cutoff_sigma = _cutoff_sigma;
  built = false;
}

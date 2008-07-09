/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/ExcisionUnpacker.h"
#include "dsp/WeightedTimeSeries.h"
#include "dsp/BitTable.h"

#include "Error.h"
#include <assert.h>

using namespace std;

//! Null constructor
dsp::ExcisionUnpacker::ExcisionUnpacker (const char* _name)
  : HistUnpacker (_name)
{
}

void dsp::ExcisionUnpacker::unpack ()
{
  if (verbose)
    cerr << "dsp::ExcisionUnpacker::unpack" << endl;;

  uint64 ndat = input->get_ndat();

  if (ndat < get_ndat_per_weight())
    return;

  const unsigned char* rawptr = input->get_rawptr();

  unsigned ndig = get_ndig ();

  // weights are used only if output is a WeightedTimeseries
  unsigned* weights = 0;
  uint64 nweights = 0;

  // the number of floating point numbers to unpack from each digitizer
  uint64 nfloat = ndat * get_ndim_per_digitizer();

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

    dig_unpack (into, from, nfloat, idig, weights, unsigned(nweights));
      
  }  // for each stream of digitized data


  if (weighted_output)
  {
    weighted_output -> mask_weights ();
    uint64 nbad = weighted_output -> get_nzero ();
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

//! Set the number of time samples used to estimate undigitized power
void dsp::ExcisionUnpacker::set_ndat_per_weight (unsigned _ndat)
{
  if (get_ndat_per_weight() != _ndat)
    built = false;

  // in two-bit correction mode, ndat_per_weight must equal nstate
  HistUnpacker::set_ndat_per_weight (_ndat);
  HistUnpacker::set_nstate (_ndat);
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

#include "dsp/WeightedTimeSeries.h"
#include "Error.h"

dsp::WeightedTimeSeries::WeightedTimeSeries()
{
  npol_weight = 1;
  nchan_weight = 1;
  ndat_per_weight = 0;

  weights = NULL;
  weight_size = 0;
  weight_subsize = 0;
}

void dsp::WeightedTimeSeries::copy_configuration (const TimeSeries* copy)
{
  TimeSeries::copy_configuration (copy);

  const WeightedTimeSeries* weighted_copy;
  weighted_copy = dynamic_cast< const WeightedTimeSeries* > (copy);

  if (!weighted_copy) {
    if (verbose)
      cerr << "dsp::WeightedTimeSeries::copy_configuration "
              "copying a non-WeightedTimeSeries" << endl;

    ndat_per_weight = 0;
    return;
  }

  if (weighted_copy == this)  {
    if (verbose) cerr << "dsp::WeightedTimeSeries::copy_configuration"
                         " copying self" << endl;
    return;
  }

  set_npol_weight  ( weighted_copy->get_npol_weight() );
  set_nchan_weight ( weighted_copy->get_nchan_weight() );
  set_ndat_per_weight ( weighted_copy->get_ndat_per_weight() );

  if (verbose) cerr << "dsp::WeightedTimeSeries::copy_configuration"
                       " resize weights" << endl;

  resize_weights ();

  copy_weights (*weighted_copy);
}

//! Set the number of time samples per weight
void dsp::WeightedTimeSeries::set_ndat_per_weight (unsigned _ndat_per_weight)
{
  ndat_per_weight = _ndat_per_weight;
}

//! Set the number of polarizations with independent weights
void dsp::WeightedTimeSeries::set_npol_weight (unsigned _npol_weight)
{
  npol_weight = _npol_weight;
}

//! Set the number of frequency channels with independent weights
void dsp::WeightedTimeSeries::set_nchan_weight (unsigned _nchan_weight)
{
  nchan_weight = _nchan_weight;
}

//! Get the number of weights
uint64 dsp::WeightedTimeSeries::get_nweights () const
{
  return get_nweights (get_ndat());
}

//! Get the number of weights
uint64 dsp::WeightedTimeSeries::get_nweights (uint64 nsamples) const
{
  if (ndat_per_weight == 0)
    return 0;
 
  if (verbose)
    cerr << "dsp::WeightedTimeSeries::get_nweights ndat_per_weight=" 
         << ndat_per_weight << " ndat=" << get_ndat() << endl;
 
  uint64 nweights = get_ndat() / ndat_per_weight;
  if (get_ndat() % ndat_per_weight)
    nweights ++;
  
  return nweights;
}

//! Allocate the space required to store nsamples time samples
/*! This method uses TimeSeries::resize to size the floating point time
  samples array, then resizes the weights array according to the nsamples
  argument as well as the ndat_per_weight, nchan_weight, and npol_weight
  attributes.
  
  \pre The ndat_per_weight, nchan_weight, and npol_weight attributes must
  be set to their desired values before this method is called.
*/
void dsp::WeightedTimeSeries::resize (uint64 nsamples)
{
  if (verbose)
    cerr << "dsp::WeightedTimeSeries::resize (" << nsamples << ")" << endl;
  
  TimeSeries::resize (nsamples);
  resize_weights ();
}

void dsp::WeightedTimeSeries::resize_weights ()
{ 
  uint64 nweights = get_nweights ();
  uint64 require = nweights * get_npol_weight() * get_nchan_weight();
  
  if (verbose)
    cerr << "dsp::WeightedTimeSeries::resize_weights nweights=" << nweights
	 << " require=" << require << " have=" << weight_size << endl;

  if (!require || require > weight_size) {
    if (verbose)
      cerr << "dsp::WeightedTimeSeries::resize_weights delete" << endl;

    if (weights) delete [] weights; weights = 0;
    weight_size = weight_subsize = 0;
  }
  
  if (!require)
    return;
  
  if (weight_size == 0) {
    if (verbose)
      cerr << "dsp::WeightedTimeSeries::resize_weights new " << require << endl;

    weights = new unsigned [require];
    weight_size = require;
  }
  
  weight_subsize = nweights;
}

//! Return pointer to the specified data block
unsigned* dsp::WeightedTimeSeries::get_weights (unsigned ichan, unsigned ipol)
{
  return weights + (ichan * npol_weight + ipol) * weight_subsize;
}

//! Return pointer to the specified data block
const unsigned*
dsp::WeightedTimeSeries::get_weights (unsigned ichan, unsigned ipol) const
{
  return weights + (ichan * npol_weight + ipol) * weight_subsize;
}


dsp::WeightedTimeSeries& 
dsp::WeightedTimeSeries::operator = (const WeightedTimeSeries& copy)
{
  TimeSeries::operator = (copy);
  copy_configuration (&copy);
  return *this;
}

void dsp::WeightedTimeSeries::copy_weights (const WeightedTimeSeries& copy)
{
  uint64 nweights = get_nweights ();

  if (verbose)
    cerr << "dsp::WeightedTimeSeries::copy_weights nweights=" << nweights
         << " nchan=" << get_nchan_weight() << " npol=" << get_npol_weight()
         << endl;
  
  for (unsigned ichan=0; ichan<get_nchan_weight(); ichan++)
    for (unsigned ipol=0; ipol<get_npol_weight(); ipol++) {
      
      unsigned* data1 = get_weights (ichan, ipol);
      const unsigned* data2 = copy.get_weights (ichan, ipol);
      
      for (uint64 iwt=0; iwt<nweights; iwt++)
        data1[iwt] = data2[iwt];
      
    }
}

dsp::WeightedTimeSeries& 
dsp::WeightedTimeSeries::operator += (const WeightedTimeSeries& add)
{
  TimeSeries::operator += (add);
  
  uint64 nweights = get_nweights ();
  
  for (unsigned ichan=0; ichan<get_nchan_weight(); ichan++)
    for (unsigned ipol=0; ipol<get_npol_weight(); ipol++) {
      
      unsigned* data1 = get_weights (ichan, ipol);
      const unsigned* data2 = add.get_weights (ichan, ipol);
      
      for (uint64 iwt=0; iwt<nweights; iwt++) {
	if (data1[iwt] == 0 || data2[iwt] == 0)
	  data1[iwt] = 0;
	else
	  data1[iwt] = (data1[iwt] + data2[iwt])/2;
      }
    }
  
  check_weights ();
  return *this;
}

void dsp::WeightedTimeSeries::zero ()
{
  TimeSeries::zero ();
  neutral_weights ();
}

//! Check that each floating point value is zeroed if weight is zero
void dsp::WeightedTimeSeries::check_weights ()
{
  mask_weights ();
}

//! Set all weights to one
void dsp::WeightedTimeSeries::neutral_weights ()
{
  for (uint64 i=0; i<weight_size; i++)
    weights[i] = 1;
}

uint64 dsp::WeightedTimeSeries::get_nzero () const
{
  uint64 nweights = get_nweights ();
  uint64 zeroes = 0;
  for (uint64 i=0; i<nweights; i++)
    if (weights[i] == 0)
      zeroes ++;

  return zeroes;
}

/*! The algorithm is written so that only two sections of the array are
  used at one time.  Should minimize the number of cache hits */
void dsp::WeightedTimeSeries::mask_weights ()
{
  uint64 nweights = get_nweights ();
  unsigned nparts = get_npol_weight() * get_nchan_weight();

  if (verbose)
    cerr << "dsp::WeightedTimeSeries::mask_weights nweights=" << nweights
	 << " nparts=" << nparts << endl;

  // collect all of the bad weights in the first array
  for (unsigned ipart=1; ipart<nparts; ipart++) {
    unsigned* wptr = weights + ipart * weight_subsize;
    for (unsigned iwt=0; iwt < nweights; iwt++) {
      if (wptr[iwt] == 0)
	weights[iwt] = 0;
    }
  }

  // distribute the bad weights to all of the arrays
  for (unsigned ipart=1; ipart<nparts; ipart++) {
    unsigned* wptr = weights + ipart * weight_subsize;
    for (unsigned iwt=0; iwt < nweights; iwt++) {
      if (weights[iwt] == 0)
	wptr[iwt] = 0;
    }
  }
}

void dsp::WeightedTimeSeries::convolve_weights (unsigned nfft, unsigned nkeep)
{
  if (ndat_per_weight >= nfft) {
    if (verbose)
      cerr << "dsp::WeightedTimeSeries::convolve_weights ndat_per_weight="
	   << ndat_per_weight << " >= nfft=" << nfft << endl;

    // the fft happens within one weight, no "convolution" required
    return;
  }

  uint64 nweights_tot = get_nweights();

  if (verbose)
    cerr << "dsp::WeightedTimeSeries::convolve_weights nfft=" << nfft
	 << " nkeep=" << nkeep << " ndat_per_weight=" << ndat_per_weight
	 << " nweights=" << nweights_tot << endl;

  double weights_per_dat = 1.0 / ndat_per_weight;

  uint64 blocks = (get_ndat()+nkeep-nfft) / nkeep;
  uint64 end_idat = blocks * nkeep;

  uint64 total_bad = 0;
  uint64 zero_start = 0;
  uint64 zero_end = 0;

  for (uint64 start_idat=0; start_idat < end_idat; start_idat += nkeep) {

    uint64 start_weight = uint64(start_idat * weights_per_dat);
    uint64 end_weight = uint64(ceil ((start_idat+nfft) * weights_per_dat));

    if (end_weight > nweights_tot)
      throw Error (InvalidState, "dsp::WeightedTimeSeries::convolve_weights",
		   "end_weight=%d > nweights=%d", end_weight, nweights_tot);

    if (verbose)
      cerr << "dsp::WeightedTimeSeries::convolve_weights start_weight="
	   << start_weight << " end_weight=" << end_weight << endl;

    uint64 zero_weights = 0;
    uint64 iweight = 0;
    for (iweight=start_weight; iweight<end_weight; iweight++)
      if (weights[iweight] == 0)
	zero_weights ++;
    
    /* If there exists bad data in the transform, the whole transform
       must be flagged as invalid; otherwise, the FFT will mix the bad
       data into the good data.  However, we cannot immediately flag
       the current transform, as this will affect the next test.
       Therefore, flag only the previous data set and set the flag for
       the next loop. */

    for (iweight=zero_start; iweight < zero_end; iweight++) {
      weights[iweight] = 0;
      total_bad ++;
    }

    if (zero_weights == 0)
      zero_start = zero_end = 0;

    else {

      if (verbose)
        cerr << "dsp::WeightedTimeSeries::convolve_weights bad weights="
	     << zero_weights << endl;

      zero_start = start_weight;
      zero_end = uint64( ceil((start_idat+nkeep) * weights_per_dat) );

      if (verbose)
	cerr << "dsp::WeightedTimeSeries::convolve_weights"
	  " flagging " << zero_end-zero_start << " bad weights" << endl;

    }

  } 

  for (uint64 iweight=zero_start; iweight < zero_end; iweight++) {
    weights[iweight] = 0;
    total_bad ++;
  }

  if (total_bad && verbose)
    cerr << "dsp::WeightedTimeSeries::convolve_weights " << get_nzero() <<
      "/" << nweights_tot << " total bad weights" << endl;

}

void dsp::WeightedTimeSeries::scrunch_weights (unsigned nscrunch)
{
  uint64 nweights_tot = get_nweights();

  if (verbose)
    cerr << "dsp::WeightedTimeSeries::scrunch_weights nscrunch=" << nscrunch
	 << " ndat_per_weight=" << ndat_per_weight
	 << " nweights=" << nweights_tot << endl;

  // the points per weight after time resolution decreases
  double points_per_weight = double(ndat_per_weight) / double(nscrunch);

  if (points_per_weight >= 1.0) {
    if (verbose)
      cerr << "dsp::WeightedTimeSeries::scrunch_weights new points_per_weight="
	   << points_per_weight << endl;
    ndat_per_weight = unsigned (points_per_weight);	
    return;
  }

  if (verbose)
    cerr << "dsp::WeightedTimeSeries::scrunch_weights"
      " scrunching to 1 point per wieght" << endl;

  // reduce the number of weights by scrunching
  uint64 new_nweights = nweights_tot / nscrunch;
  uint64 extra = nweights_tot % nscrunch;

  if (extra)
    new_nweights ++;

  for (uint64 iwt=0; iwt < new_nweights; iwt++) {
    
    unsigned* indi_weight = weights + iwt * nscrunch;
    
    if ((iwt+1)*nscrunch > nweights_tot)
      nscrunch = extra;
    
    for (unsigned ivt=0; ivt < nscrunch; ivt++) {
      if (*indi_weight == 0) {
	weights[iwt] = 0;
	break;
      }
      else if (ivt == 0)
	weights[iwt] = *indi_weight;
      else
	weights[iwt] += *indi_weight;

      indi_weight ++;
    }

    weights[iwt] /= nscrunch;
  }

  ndat_per_weight = 1;

}

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
  if (ndat_per_weight == 0)
    return 0;
  
  uint64 nweights = ndat / ndat_per_weight;
  if (ndat % ndat_per_weight)
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
  
  uint64 nweights = get_nweights ();
  uint64 require = nweights * get_npol_weight() * get_nchan_weight();
  
  if (verbose)
    cerr << "dsp::WeightedTimeSeries::resize nweights=" << nweights
	 << " require=" << require << endl;

  if (!require || require > weight_size) {
    if (weights) delete [] weights; weights = 0;
    weight_size = weight_subsize = 0;
  }
  
  if (!require)
    return;
  
  if (weight_size == 0) {
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
  if (this == &copy)
    return *this;
  
  set_npol_weight  ( copy.get_npol_weight() );
  set_nchan_weight ( copy.get_nchan_weight() );
  set_ndat_per_weight ( copy.get_ndat_per_weight() );
  
  TimeSeries::operator = (copy);
  
  uint64 nweights = get_nweights ();
  
  for (unsigned ichan=0; ichan<get_nchan_weight(); ichan++)
    for (unsigned ipol=0; ipol<get_npol_weight(); ipol++) {
      
      unsigned* data1 = get_weights (ichan, ipol);
      const unsigned* data2 = copy.get_weights (ichan, ipol);
      
      for (uint64 iwt=0; iwt<nweights; iwt++)
        data1[iwt] = data2[iwt];
      
    }
  
  return *this;
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


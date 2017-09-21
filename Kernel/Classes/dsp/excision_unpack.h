//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/excision_unpack.h

#ifndef __ExcisionUnpacker_excision_unpack_h
#define __ExcisionUnpacker_excision_unpack_h

#include "dsp/ExcisionUnpacker.h"

#include <iostream>
#include <assert.h>

using namespace std;

template<class U, class Iter>
  void dsp::ExcisionUnpacker::excision_unpack (U& unpack,
					       Iter& input_data,
					       float* output_data,
					       uint64_t nfloat,
					       unsigned long* hist,
					       unsigned* weights,
					       unsigned nweights)
{
#ifndef _DEBUG
  if (verbose)
#endif
    cerr << "dsp::ExcisionUnpacker::excision_unpack in=" << input_data.ptr()
	 << " out=" << output_data << " nfloat=" << nfloat << "\n\t"
	 << " hist=" << hist << " weights=" << weights 
	 << " nweights=" << nweights << endl;

  const unsigned ndim_dig = get_ndim_per_digitizer();
  const unsigned nfloat_per_weight = get_ndat_per_weight() * ndim_dig;
  const unsigned long n_weights = nfloat / nfloat_per_weight;

  assert (nfloat % nfloat_per_weight == 0);

  if (weights && n_weights > nweights)
    throw Error (InvalidParam, "dsp::ExcisionUnpacker::excision_unpack",
		 "weights array size=%d < nweights=%d", nweights, n_weights);

  const unsigned output_incr = get_output_incr ();

  const unsigned hist_size = get_ndat_per_weight();

  unsigned n_low = 0;

  for (unsigned long wt=0; wt<n_weights; wt++)
  {
#ifdef _DEBUG
    cerr << wt << " ";
#endif

    // the prepare method should not modify the iterator
    unpack.prepare (input_data, nfloat_per_weight);

    // the unpack method should modify the iterator
    unpack.unpack (input_data, nfloat_per_weight,
		   output_data, output_incr, n_low);

    // if data are complex, quickly divide n_low by two
    if (ndim_dig == 2)
      n_low >>= 1;

    if (hist && n_low < hist_size)
      hist [n_low] ++;

    /*
      Data are flagged as bad and set to zero if:

      1) the unpacker says it is bad;
      2) the number of low voltage states is outside thresholds; or
      3) this section of data has been previously flagged bad 
         (e.g. due to bad data in the other polarization).
    */

    if ( unpack.bad ||
	 n_low<nlow_min || n_low>nlow_max ||
	 (weights && weights[wt] == 0) )
    {
#ifdef _DEBUG2
      cerr << "w[" << wt << "]=0 ";
#endif
      if (weights)
        weights[wt] = 0;
      
      // reduce the risk of other functions accessing un-initialized 
      // segments of the array
      for (unsigned ifloat=0; ifloat<nfloat_per_weight; ifloat++)
	output_data [ifloat * output_incr] = 0.0;
    }

    output_data += nfloat_per_weight * output_incr;
  }

#ifdef _DEBUG
  cerr << "DONE!" << endl;
#endif
  
}

#endif

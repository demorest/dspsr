/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/ExcisionBitUnpacker.h"
#include "dsp/Input.h"
#include "dsp/BitTable.h"

using namespace std;

//! Null constructor
dsp::ExcisionBitUnpacker::ExcisionBitUnpacker (const char* name)
  : ExcisionUnpacker (name)
{
  minimum_ndat_per_weight = 512;
}

//! Set the BitUnpacker to be used to unpack data
void dsp::ExcisionBitUnpacker::set_unpacker (BitUnpacker* u)
{
  unpacker = u;
}

unsigned dsp::ExcisionBitUnpacker::get_input_stride (unsigned nfloat) const
{
  return input->get_nbytes (nfloat) * get_input_incr ();
}

//! Unpack a single digitized stream from raw into data
void dsp::ExcisionBitUnpacker::dig_unpack (float* output_data,
					   const unsigned char* input_data, 
					   uint64 nfloat,
					   unsigned digitizer,
					   unsigned* weights,
					   unsigned nweights)
{
  if (!get_ndat_per_weight())
  {
    unsigned points_per_weight = input->get_loader()->get_resolution();

    cerr << "dsp::ExcisionBitUnpacker::dig_unpack resolution="
	 << points_per_weight << endl;

    if (points_per_weight < minimum_ndat_per_weight)
    {
      unsigned times = minimum_ndat_per_weight / points_per_weight;
      if (minimum_ndat_per_weight % points_per_weight)
	times ++;
      points_per_weight *= times;
    }

    cerr << "dsp::ExcisionBitUnpacker::dig_unpack ndat_per_weight="
	 << points_per_weight << endl;

    set_ndat_per_weight( points_per_weight );
  }

  // number of floating-point samples per byte
  //const unsigned samples_per_byte = table->get_values_per_byte();

  // histogram to in which counts are kept
  unsigned long* hist = 0;
  if (keep_histogram)
    hist = get_histogram (digitizer);

  bool data_are_complex = get_ndim_per_digitizer() == 2;

  const unsigned input_stride = get_input_stride (nfloat);
  const unsigned input_incr = get_input_incr ();
  const unsigned output_incr = get_output_incr ();

  const unsigned nfloat_per_weight
    = get_ndat_per_weight() * get_ndim_per_digitizer();

  double f_weights = double(nfloat) / double(nfloat_per_weight);
  unsigned long n_weights = (unsigned long) ceil (f_weights);

  assert (n_weights*nfloat_per_weight >= nfloat);

  if (weights && n_weights > nweights)
    throw Error (InvalidParam, "dsp::TwoBitCorrection::dig_unpack",
		 "weights array size=%d < nweights=%d", nweights, n_weights);
  
  for (unsigned wt=0; wt<n_weights; wt++)
  {
    vector<unsigned long> temp_hist (BitTable::unique_bytes, 0);

    unpacker->unpack (nfloat,
		      input_data, input_incr,
		      output_data, output_incr,
		      &(temp_hist[0]));

    unsigned long n_lo = 0;

    bool all_zero = (temp_hist[0] == nfloat);

    if (!all_zero)
      for (unsigned i=0; i<BitTable::unique_bytes; i++)
	n_lo += temp_hist[i] * number_of_low_states[i];

    // if data are complex, quickly divide n_lo by two
    if (data_are_complex)
      n_lo >>= 1;

    if ( !all_zero && hist && n_lo < nlow_max )
      hist [n_lo] ++;

    if ( all_zero || n_lo < nlow_min || n_lo > nlow_max 
	 || (weights && weights[wt] == 0) )
      {
	if (weights)
	  weights[wt] = 0;

	for (unsigned ifloat = 0; ifloat < nfloat; ifloat++)
	  output_data[ifloat * output_incr] = 0.0;
      }

    output_data += nfloat * output_incr;
    
    input_data += input_stride;
    
  }
  
}

#include "dsp/SubByteTwoBitCorrection.h"
#include "dsp/TwoBitTable.h"

template<class Mask>
void dsp::SubByteTwoBitCorrection::dig_unpack (Mask& mask,
					       float* output_data,
					       const unsigned char* input_data,
					       uint64 ndat,
					       unsigned digitizer,
					       unsigned* weights,
					       unsigned nweights)
{
  if (verbose)
    cerr << "dsp::SubByteTwoBitCorrection::dig_unpack" << endl;

  if (!values)
    throw Error (InvalidState, "dsp::SubByteTwoBitCorrection::dig_unpack",
		 "not built");

  const unsigned ndig = get_ndig_per_byte();
  const unsigned samples_per_byte = TwoBitTable::vals_per_byte / ndig;

  if (ndig < 2)
    throw Error (InvalidState, "dsp::SubByteTwoBitCorrection::dig_unpack",
		 "number of digitizers per byte = %d must be > 1", ndig);



  const unsigned char* input_data_ptr = input_data;

  float* output_data_ptr = output_data;

  const unsigned input_incr = get_input_incr ();
  const unsigned output_incr = get_output_incr ();
  const unsigned nsample = get_nsample ();

  // although I and Q are switched here, the histogram is queried as expected
  unsigned long*  hist = 0;
  if (keep_histogram)
    hist = get_histogram (digitizer);

  unsigned required_nweights = (unsigned) ceil (float(ndat)/float(nsample));

  if (weights)  {
    if (verbose) cerr << "dsp::SubByteTwoBitCorrection::dig_unpack nweights=" 
		      << nweights << endl;

    if (required_nweights > nweights)
      throw Error (InvalidParam, "dsp::SubByteTwoBitCorrection::dig_unpack",
		   "weights array size=%d < nweights=%d",
		   nweights, required_nweights);

  }

  nweights = required_nweights;
  uint64 points_left = ndat;

  uint64 points = nsample;

  for (unsigned wt=0; wt<nweights; wt++) {

    if (points > points_left)
      points = points_left;

    uint64 pt = 0;

    // retrieve the next points values from the 2bit data
    while (pt < points) {
      for (unsigned isamp=0; isamp<samples_per_byte; isamp++) {
	values[pt] = mask.twobit (*input_data_ptr, isamp);
	pt++;
      }
      input_data_ptr += input_incr;
    }

    // calculate the weight based on the last nsample pts
    unsigned n_in = 0;
    for (pt=0; pt<nsample; pt++)
      n_in += lovoltage [values[pt]];

    if (hist)
      hist [n_in] ++;

    // test if the number of low voltage states is outside the
    // acceptable limit or if this section of data has been previously
    // flagged bad (for example, due to bad data in the other polarization)
    if ( n_in<n_min || n_in>n_max || (weights && weights[wt]==0) ) {

#ifdef _DEBUG
      cerr << "w[" << wt << "]=0 ";
#endif

      if (weights)
        weights[wt] = 0;

      // reduce the risk of other functions accessing un-initialized 
      // segments of the array
      for (pt=0; pt<points; pt++) {
	*output_data_ptr = 0.0;
	output_data_ptr += output_incr;
      }

    }

    else {
      float* corrected = &(dls_lookup[0]) + (n_in-n_min) * 4;
      for (pt=0; pt<points; pt++) {
	*output_data_ptr = corrected [values[pt]];
	output_data_ptr += output_incr;
      }
      if (weights)
	weights[wt] += n_in;
    }

    points_left -= points;
  }


}

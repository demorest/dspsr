#include "dsp/SubByteTwoBitCorrection.h"
#include "dsp/TwoBitTable.h"

#include "genutil.h"

dsp::SubByteTwoBitCorrection::SubByteTwoBitCorrection (const char* name)
  : TwoBitCorrection (name)
{
  values = 0;
}

dsp::SubByteTwoBitCorrection::~SubByteTwoBitCorrection ()
{
  destroy ();
}

/*! By default, both polarizations are output in one byte */
unsigned dsp::SubByteTwoBitCorrection::get_ndig_per_byte () const
{ 
  return 2;
}

/*! By default, the data is not interleaved byte by byte */
unsigned dsp::SubByteTwoBitCorrection::get_input_offset (unsigned idig) const
{
  return 0;
}

/*! By default, the data is not interleaved byte by byte */
unsigned dsp::SubByteTwoBitCorrection::get_input_incr () const 
{
  return 1;
}

/*! By default, MSB y1 x1 y0 x0 LSB */
unsigned dsp::SubByteTwoBitCorrection::get_shift (unsigned idig, unsigned samp) const
{
  return (idig + samp * 2) * 2;
}

void dsp::SubByteTwoBitCorrection::dig_unpack (float* output_data,
					       const unsigned char* input_data, 
					       uint64 ndat,
					       unsigned digitizer,
					       unsigned* weights,
					       unsigned nweights)
{
  if (!values)
    throw Error (InvalidState, "dsp::SubByteTwoBitCorrection::dig_unpack",
		 "not built");

  unsigned char mask2 = 0x03;

  unsigned ndig = get_ndig_per_byte();
  unsigned samples_per_byte = TwoBitTable::vals_per_byte / ndig;

  if (ndig < 2)
    throw Error (InvalidState, "dsp::SubByteTwoBitCorrection::dig_unpack",
		 "number of digitizers per byte = %d must be > 1", ndig);

  unsigned isamp=0;
  unsigned shift[2];

  for (unsigned isamp=0; isamp<samples_per_byte; isamp++)
    shift[isamp] = get_shift (digitizer, isamp);

  const unsigned char* input_data_ptr = input_data;

  float* output_data_ptr = output_data;
  unsigned output_incr = get_output_incr ();

  // although I and Q are switched here, the histogram is queried as expected
  unsigned long*  hist = 0;
  if (keep_histogram)
    hist = &(histograms[digitizer][0]);

  unsigned long n_weights = (unsigned long) ceil (float(ndat)/float(nsample));

  if (weights && n_weights > nweights)
    throw Error (InvalidParam, "dsp::SubByteTwoBitCorrection::dig_unpack",
		 "weights array size=%d < number of weights=%d", nweights, n_weights);

  uint64 points_left = ndat;

  for (unsigned long wt=0; wt<n_weights; wt++) {

    unsigned points = nsample;
    if (points > points_left)
      points = points_left;

    unsigned pt   = 0;

    // retrieve the next points values from the 2bit data
    while (pt < points) {
      for (isamp=0; isamp<samples_per_byte; isamp++) {
	values[pt] = ((*input_data_ptr) >> shift[isamp]) & mask2;
	pt++;
      }
      input_data_ptr ++;
    }

    // calculate the weight based on the last nsample pts
    unsigned n_in = 0;
    for (pt=0; pt<nsample; pt++)
      n_in += lovoltage [values[pt]];

    if (hist)
      hist [n_in] ++;

    if ( (weights && weights[wt]==0) || n_in<n_min || n_in>n_max ) {

#ifdef _DEBUG
      cerr << "iq:w[" << wt << "]=0 ";
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

void dsp::SubByteTwoBitCorrection::build ()
{
  if (verbose)
    cerr << "dsp::SubByteTwoBitCorrection::build" << endl;

  // delete the old space
  SubByteTwoBitCorrection::destroy();

  // setup the lookup table
  TwoBitCorrection::build ();

  // create the new space
  values = new unsigned char [nsample];
}

void dsp::SubByteTwoBitCorrection::nlo_build ()
{
  if (verbose)
    cerr << "dsp::SubByteTwoBitCorrection::nlo_build" << endl;

  float fourvals [TwoBitTable::vals_per_byte];
  float lo_valsq = 1.0;

  // flatten the table again (precision errors cause mismatch of lo_valsq)
  table->set_lo_val (1.0);
  table->four_vals (fourvals);

  for (unsigned ifv=0; ifv<TwoBitTable::vals_per_byte; ifv++)
    if (fourvals[ifv]*fourvals[ifv] == lo_valsq)
      lovoltage[ifv] = 1;
    else
      lovoltage[ifv] = 0;
}

void dsp::SubByteTwoBitCorrection::destroy ()
{
  if (values != NULL) delete [] values;  values = NULL;
}


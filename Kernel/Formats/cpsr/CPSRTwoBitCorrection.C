#include "dsp/CPSRTwoBitCorrection.h"
#include "dsp/TwoBitTable.h"
#include "dsp/WeightedTimeSeries.h"

#include "genutil.h"

bool dsp::CPSRTwoBitCorrection::matches (const Observation* observation)
{
  return observation->get_machine() == "CPSR";
}

//! Null constructor
dsp::CPSRTwoBitCorrection::CPSRTwoBitCorrection ()
  : TwoBitCorrection ("CPSRTwoBitCorrection")
{
  values = 0;

  nchannel = 4;
  channels_per_byte = 4;
  table = new TwoBitTable (TwoBitTable::OffsetBinary);
}

void dsp::CPSRTwoBitCorrection::unpack ()
{
  if (input->get_npol() != 2)
    throw Error (InvalidParam, "dsp::CPSRTwoBitCorrection::unpack",
		 "input not dual-poln");

  if (input->get_ndim() != 2)
    throw Error (InvalidParam, "dsp::CPSRTwoBitCorrection::unpack",
		 "input not quadrature sampled");

  uint64 ndat = input->get_ndat();
  const unsigned char* rawptr = input->get_rawptr();

  // weights are used only if output is a WeightedTimeseries
  unsigned* weights = 0;

  for (int ipol=0; ipol<2; ipol++) {

    // for each of in-phase and quadrature components
    for (int iq=0; iq<2; iq++) {
      
      int channel = ipol * 2 + iq;

      float* unpackinto = output->get_datptr(0,ipol) + iq;
      
      if (verbose)
	fprintf (stderr, "dsp::CPSRTwoBitCorrection::unpack"
		 " into data[%d]=%p (chan:%d)\n", ipol, unpackinto, channel);

      // if the output TimeSeries is a weighted output, use its weights array
      if (weighted_output)
	weights = weighted_output -> get_weights (0, ipol);

      iq_unpack (unpackinto, rawptr, ndat, channel, weights);
      
    }  // for each of I and Q

#if 0
    voltage_unpack ((unsigned char*) in_bs.data, (void *) unpackinto,
		    ipol*in_bs.riq, in_bs.npol*in_bs.riq,
		    in_bs.nbit, fs->ndat,
		    0 /* SSB */, 1 /*VOLTS*/, 1 /*FLOAT*/);
#endif

  }  // for each polarization

}

void dsp::CPSRTwoBitCorrection::iq_unpack (float* outdata,
					   const unsigned char* raw,
					   uint64 ndat, 
					   unsigned channel,
					   unsigned* weights)
{
  if (!values)
    throw Error (InvalidState, "dsp::CPSRTwoBitCorrection::iq_unpack",
		 "not built");

  if (nsample < 10)
    throw Error (InvalidState, "dsp::CPSRTwoBitCorrection::iq_unpack",
		 "invalid nsample %d", nsample);

  if (channel < 0 || channel >= nchannel)
    throw Error (InvalidParam, "dsp::CPSRTwoBitCorrection::iq_unpack",
		 "invalid channel %d", channel);

  if (ndat < nsample)
    throw Error (InvalidParam, "dsp::CPSRTwoBitCorrection::iq_unpack",
		 "ndat=%d < nsample=%d", ndat, nsample);

  static int ones [4] = { 0, 1, 1, 0 };

  unsigned char mask2 = 0x03;
  int shift;

  // switch I and Q
  unsigned newchan = channel +1 - 2*(channel % 2);
  // int newchan = channel;
  shift = (4 - newchan - 1) * 2;

  const unsigned char* rawptr = raw;
  float* datptr = outdata;

  // although I and Q are switched here, the histogram is queried as expected
  unsigned long*  hist = 0;
  if (keep_histogram)
    hist = &(histograms[channel][0]);

  //fprintf (stderr, "tbc: chan: %d\n", newchan);
  unsigned nsuccess = 0;

  unsigned long n_weights = (unsigned long) ceil (float(ndat)/float(nsample));

  uint64 points_left = ndat;

  // these aren't true totals, just accurate enough to get the fractional
  // points and judge against the tolerance towards the varying variance
  unsigned long total_ones = 0;
  unsigned long total_points = 0;

  for (unsigned long wt=0; wt<n_weights; wt++) {
    unsigned n_in = 0;
    unsigned pt   = 0;

    unsigned points = nsample;
    if (points > points_left)
      points = points_left;

    // retrieve the next points values from the 2bit data
    for (pt=0; pt<points; pt++) {
      values[pt] = ((*rawptr) >> shift) & mask2;
      rawptr ++;
    }

    // calculate the weight based on the last nsample pts
    for (pt=0; pt<nsample; pt++) {
      n_in += ones [values[pt]];
    }
    total_points += nsample;
    total_ones += n_in;
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
	*datptr = 0.0;
	datptr += 2;
      }
    }

    else {
      float* corrected = &(dls_lookup[0]) + (n_in-n_min) * 4;
      for (pt=0; pt<points; pt++) {
	*datptr = corrected [values[pt]];

#ifdef _DEBUG
	if (*datptr > 5 || *datptr < -5)
	  cerr << "b:" << *datptr << " n:" << n_in << " max:" << n_max << " min:" << n_min << endl;
#endif

	datptr += 2;
      }
      if (weights)
	weights[wt] += n_in;
      nsuccess ++;
    }

    points_left -= points;
  }

#ifdef _DEBUG
  fprintf (stderr, "iq_unpack: exits\n");
  fflush (stderr);
#endif

}

void dsp::CPSRTwoBitCorrection::build ()
{
  if (verbose)
    cerr << "dsp::CPSRTwoBitCorrection::build" << endl;

  // delete the old space
  CPSRTwoBitCorrection::destroy();

  // setup the lookup table
  TwoBitCorrection::build ();

  // create the new space
  values = new unsigned char [nsample];
}

void dsp::CPSRTwoBitCorrection::destroy ()
{
  if (values != NULL) delete [] values;  values = NULL;
}


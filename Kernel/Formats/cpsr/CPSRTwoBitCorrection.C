#include "CPSRTwoBitCorrection.h"
#include "Timeseries.h"
#include "genutil.h"

//! Null constructor
dsp::CPSRTwoBitCorrection::CPSRTwoBitCorrection (int _nsample,
						 float _cutoff_sigma)
  : TwoBitCorrection ("CPSRTwoBitCorrection", outofplace)
{
  values = 0;

  nchannel = 4;
  type = TwoBitTable::OffsetBinary;
  channels_per_byte = 4;

  nsample = _nsample;
  cutoff_sigma = _cutoff_sigma;
}

void dsp::CPSRTwoBitCorrection::unpack ()
{
  if (input->get_npol() != 2)
    throw_str ("TwoBitCorrection::operation input not dual-poln");

  if (input->get_ndim() != 2)
    throw_str ("TwoBitCorrection::operation input not quadrature sampled");

  int64 ndat = input->get_ndat();
  const unsigned char* rawptr = input->get_rawptr();

  for (int ipol=0; ipol<2; ipol++) {

    // for each of in-phase and quadrature components
    for (int iq=0; iq<2; iq++) {
      
      int channel = ipol * 2 + iq;

      float* unpackinto = output->get_datptr(0,ipol) + iq;
      
      if (verbose)
	fprintf (stderr, "CPSRTwoBitCorrection::unpack"
		 " into data[%d]=%p (chan:%d)\n", ipol, unpackinto, channel);
      
      iq_unpack (unpackinto, rawptr, ndat, channel, NULL);
      
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
					   int64 ndat, 
					   int channel, int* weights)
{
  if (!values)
    throw_str ("CPSRTwoBitCorrection::iq_unpack not built");

  if (channel < 0 || channel >= nchannel)
    throw_str ("CPSRTwoBitCorrection::iq_unpack invalid channel %d", channel);

  if (nsample < 10)
    throw_str ("CPSRTwoBitCorrection::iq_unpack invalid nsample %d", nsample);

  if (ndat < nsample)
    throw_str ("CPSRTwoBitCorrection::iq_unpack ndat=%d < nsample=%d",
	       ndat, nsample);

  static int ones [4] = { 0, 1, 1, 0 };

  unsigned char mask2 = 0x03;
  int shift;

  // switch I and Q
  int newchan = channel +1 - 2*(channel % 2);
  // int newchan = channel;
  shift = (4 - newchan - 1) * 2;

  const unsigned char* rawptr = raw;
  float* datptr = outdata;

  // although I and Q are switched here, the histogram is queried as expected
  unsigned long*  hist = 0;
  if (keep_histogram)
    hist = histograms[channel].begin();

  //fprintf (stderr, "tbc: chan: %d\n", newchan);
  int nsuccess = 0;

  unsigned long n_weights = (unsigned long) ceil (float(ndat)/float(nsample));

  int64 points_left = ndat;

  // these aren't true totals, just accurate enough to get the fractional
  // points and judge against the tolerance towards the varying variance
  unsigned long total_ones = 0;
  unsigned long total_points = 0;

  for (unsigned long wt=0; wt<n_weights; wt++) {
    int n_in = 0;
    int pt   = 0;

    int points = nsample;
    if (points > (int) points_left)
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

    if (weights && (weights[wt]==0 || n_in<n_min || n_in>n_max)) {

      // cerr << "iq:weight[" << wt << "]=0" << endl;

      weights[wt] = 0;
      // reduce the risk of other functions accessing un-initialized 
      // segments of the array
      for (pt=0; pt<points; pt++) {
	*datptr = 0.0;
	datptr += 2;
      }
    }

    else {
      float* corrected = dls_lookup.begin() + (n_in-n_min) * 4;
      for (pt=0; pt<points; pt++) {
	*datptr = corrected [values[pt]];
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


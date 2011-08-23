/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/TFPFilterbank.h"
#include "FTransform.h"

using namespace std;

dsp::TFPFilterbank::TFPFilterbank () : Filterbank ("TFPFilterbank", anyplace)
{
  pscrunch = false;
}

/*
  These are preparations that could be performed once at the start of
  the data processing
*/
void dsp::TFPFilterbank::custom_prepare ()
{
  output->set_order( TimeSeries::OrderTFP );
}

void dsp::TFPFilterbank::filterbank ()
{
  const uint64_t ndat = input->get_ndat();
  const unsigned npol = input->get_npol();
  const unsigned input_ichan = 0;

  const unsigned padding = (pscrunch || npol==1) ? 1 : 2;

  if (verbose)
    cerr << "dsp::TFPFilterbank::filterbank input ndat=" << ndat << endl;

  // number of FFTs
  uint64_t npart = ndat / nsamp_fft;

  float* outdat = output->get_dattfp ();

  for (unsigned ipol=0; ipol < npol; ipol++)
  {
    const float* indat = input->get_datptr (input_ichan, ipol);

    for (uint64_t ipart=0; ipart < npart; ipart++)
    {
      if (input->get_state() == Signal::Nyquist)
        forward->frc1d (nsamp_fft, outdat, indat);
      else
        forward->fcc1d (nsamp_fft, outdat, indat);

      for (unsigned ichan=0; ichan < nchan; ichan++)
      {
	// Re squared
	outdat[ichan*padding] = outdat[ichan*2] * outdat[ichan*2];
	// plus Im squared
	outdat[ichan*padding] += outdat[ichan*2+1] * outdat[ichan*2+1];
      }

      outdat += nchan*padding;
      indat += nchan*2;
    }
  }

  const uint64_t nfloat = npart * nchan;

  if (npol == 2)
  {
    /* the data are now in PTF order, whereas TFP is desired. */

    outdat = output->get_dattfp ();

    if (pscrunch)
    {
      if (verbose)
        cerr << "dsp::TFPFilterbank::filterbank pscrunching" << endl;

      for (uint64_t ifloat=0; ifloat < nfloat; ifloat++)
        outdat[ifloat] += outdat[ifloat+nfloat];

      output->set_npol (1);
      output->set_state (Signal::Intensity);
    }
    else
    {
      if (verbose)
        cerr << "dsp::TFPFilterbank::filterbank interleaving" << endl;

      // set Im[p0] = Re[p1]

      for (uint64_t ifloat=0; ifloat < nfloat; ifloat++)
         outdat[ifloat*2+1] = outdat[(ifloat+nfloat)*2];

      output->set_state (Signal::PPQQ);
    }
  }
  
  output->set_ndim (1);
}


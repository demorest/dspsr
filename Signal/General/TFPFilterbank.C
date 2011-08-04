/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/TFPFilterbank.h"
#include "FTransform.h"

using namespace std;

// #define _DEBUG 1

dsp::TFPFilterbank::TFPFilterbank () : Filterbank ("TFPFilterbank", anyplace)
{
  pscrunch = 1;
  tscrunch = 0;
  debugd = 0;
}

void dsp::TFPFilterbank::set_engine (Engine* _engine)
{
  engine = _engine;
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

  if (verbose)
    cerr << "dsp::TFPFilterbank::filterbank input ndat=" << ndat << endl;

  // number of FFTs
  uint64_t npart = ndat / nsamp_fft;
  const unsigned long nfloat = nsamp_fft * input->get_ndim();

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

      outdat += nfloat;
      indat += nfloat;
    }
  }

  if (debugd < 1) 
  {  
    cerr << "dsp::TFPFilterbank::filterbank tscrunch=" << tscrunch << ", pscrunch=" << pscrunch << 
            ", ndat=" << ndat << ", samp_fft=" << nsamp_fft << ", npart=" << npart << endl;
    debugd++;
  }

  if (npol != 2) 
    cerr << "dsp::TFPFilterbank::filterbank no real support for npol != 2" << endl;

  if (npol == 2)
  {

    // square law detect, keeping data in TPF order

    /* the data are now in TPF order, whereas TFP is desired.
       so square law detect, then pack p1 into the p0 holes */
    uint64_t nfloat = npart * npol * nchan;
    outdat = output->get_dattfp ();

    if (verbose)
      cerr << "dsp::TFPFilterbank::filterbank detecting" << endl;

    for (uint64_t ifloat=0; ifloat < nfloat; ifloat++)
    {
      // Re squared
      outdat[ifloat*2] *= outdat[ifloat*2];
      // plus Im squared
      outdat[ifloat*2] += outdat[ifloat*2+1] * outdat[ifloat*2+1];
    }

    // tscrunch by the required amount, keeping data in TPF order
    if (tscrunch) 
    {

      if (verbose)
        cerr << "dsp::TFPFilterbank::filterbank tscrunching" << endl;

      outdat = output->get_dattfp ();
      float * indat = outdat;

      // number of scrunched bins
      const uint64_t nbins = npart / tscrunch;

      for (uint64_t ibin=0; ibin < nbins; ibin++)
      {
        for (uint64_t iscrunch=0; iscrunch < tscrunch; iscrunch++)
        {
          for (uint64_t ichan=0; ichan < nchan; ichan+=2)
          {
            if (iscrunch == 0)
              outdat[ichan] = indat[ichan];
            else
              outdat[ichan] += indat[ichan];
          }
          
          indat += nchan;
        }
        outdat += nchan;
      }

      npart /= tscrunch;
      output->set_ndat (npart);

    }

    if (pscrunch)
    {
      if (verbose)
        cerr << "dsp::TFPFilterbank::filterbank pscrunching" << endl;

      nfloat = npart * nchan;
      for (uint64_t ifloat=0; ifloat < nfloat; ifloat++)
      {
        outdat[ifloat] = outdat[ifloat*2] + outdat[ifloat*2+nfloat];
      }

      output->set_npol (1);
      output->set_state (Signal::Intensity);
    }
    else
    {

      if (verbose)
        cerr << "dsp::TFPFilterbank::filterbank interleaving" << endl;

      nfloat = npart * nchan;

      for (uint64_t ifloat=0; ifloat < nfloat; ifloat++)
      {
        // set Im[p0] = Re[p1]
        outdat[ifloat*2+1] = outdat[ifloat*2+nfloat];
      }
      output->set_state (Signal::PPQQ);

    }
    output->set_ndim (1);

  }
}

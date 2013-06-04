/***************************************************************************
 *
 *   Copyright (C) 2012 by Willem van Straten and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/FilterbankEngine.h"
#include "dsp/SpatialFilterbank.h"
#include "FTransform.h"

using namespace std;

dsp::SpatialFilterbank::SpatialFilterbank () : Filterbank ("SpatialFilterbank", anyplace)
{

}

/*
  These are preparations that could be performed once at the start of
  the data processing
*/
void dsp::SpatialFilterbank::custom_prepare ()
{
  output->set_order (TimeSeries::OrderTFP);
  output->set_state (Signal::Analytic);
}

void dsp::SpatialFilterbank::filterbank ()
{
  if (verbose)
    cerr << "dsp::SpatialFilterbank::filterbank()" << endl;

  const uint64_t ndat = input->get_ndat();
  const unsigned npol = input->get_npol();
  const unsigned input_ichan = 0;

  if (verbose)
    cerr << "dsp::SpatialFilterbank::filterbank input ndat=" << ndat << endl;

  // number of FFTs
  if (verbose)
  {
    cerr << "dsp::SpatialFilterbank::filterbank input npart=" << npart << endl;
    cerr << "dsp::SpatialFilterbank::filterbank input nchan=" << nchan << endl;
    cerr << "dsp::SpatialFilterbank::filterbank input npol=" << npol << endl;
    cerr << "dsp::SpatialFilterbank::filterbank nsamp_step=" << nsamp_step << endl;
  }

  uint64_t in_step = nsamp_step * input->get_ndim();
  uint64_t out_step = nsamp_step * npol * 2;
  if (verbose)
  {
    cerr << "dsp::SpatialFilterbank::filterbank in_step=" << in_step << endl;
    cerr << "dsp::SpatialFilterbank::filterbank out_step=" << out_step << endl;
  }

  if (engine)
  {
    unsigned nkeep = freq_res - nfilt_tot;

    //cerr << "dsp::SpatialFilterbank::filterbank npart=" << npart << endl;
    //cerr << "dsp::SpatialFilterbank::filterbank engine->configure (" << nchan << ", " << nfilt_pos << ", " << freq_res << ", " << nkeep << ")" << endl;
    engine->configure (nchan, nfilt_pos, freq_res, nkeep);

    //cerr << "dsp::SpatialFilterbank::filterbank engine->perform()" << endl;
    engine->perform(input, output, npart, in_step, out_step);

    if (Operation::record_time)
    {
      engine->finish();
    }
    return;
  }

  // CPU version
  float* time_dom_ptr = NULL;

  for (unsigned input_ichan=0; input_ichan<input->get_nchan(); input_ichan++)
  {
    for (unsigned ipol=0; ipol < npol; ipol++)
    {
      const float* indat = input->get_datptr (input_ichan, ipol);
      // TODO FIX THIS!!!
      float *outdat      = output->get_dattfp ();

      for (uint64_t ipart=0; ipart < npart; ipart++)
      {
        if (input->get_state() == Signal::Nyquist)
          forward->frc1d (nsamp_fft, outdat, indat);
        else
          forward->fcc1d (nsamp_fft, outdat, indat);

        outdat += nchan * 2 * npol;
        indat += nchan * 2;
      }
    }
  }
}

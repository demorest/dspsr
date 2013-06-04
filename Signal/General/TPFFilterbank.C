/***************************************************************************
 *
 *   Copyright (C) 2012 by Willem van Straten and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/FilterbankEngine.h"
#include "dsp/TPFFilterbank.h"
#include "FTransform.h"

using namespace std;

dsp::TPFFilterbank::TPFFilterbank () : Filterbank ("TPFFilterbank", anyplace)
{

}

/*
  These are preparations that could be performed once at the start of
  the data processing
*/
void dsp::TPFFilterbank::custom_prepare ()
{
  output->set_order (TimeSeries::OrderTPF);
  output->set_state (Signal::Analytic);
}


void set_pointers (dsp::Filterbank::Engine* engine, dsp::TimeSeries* output,
                   unsigned ipol = 0)
{   
  //engine->nchan = output->get_nchan();
  //engine->output = output->get_dattpf ();
  //engine->output_span = output->get_datptr (1, ipol) - output->get_datptr (0, ipol);
} 

void dsp::TPFFilterbank::filterbank ()
{
  if (verbose)
    cerr << "dsp::TPFFilterbank::filterbank()" << endl;

  const uint64_t ndat = input->get_ndat();
  const unsigned npol = input->get_npol();
  const unsigned input_ichan = 0;

  if (verbose)
    cerr << "dsp::TPFFilterbank::filterbank input ndat=" << ndat << endl;

  // number of FFTs
  if (verbose)
  {
    cerr << "dsp::TPFFilterbank::filterbank input npart=" << npart << endl;
    cerr << "dsp::TPFFilterbank::filterbank input nchan=" << nchan << endl;
    cerr << "dsp::TPFFilterbank::filterbank input npol=" << npol << endl;
    cerr << "dsp::TPFFilterbank::filterbank nsamp_step=" << nsamp_step << endl;
  }

  uint64_t in_step = nsamp_step * input->get_ndim();
  uint64_t out_step = nchan * npol * 2;
  if (verbose)
  {
    cerr << "dsp::TPFFilterbank::filterbank in_step=" << in_step << endl;
    cerr << "dsp::TPFFilterbank::filterbank out_step=" << out_step << endl;
  }

  if (engine)
  {
    engine->perform(input, output, npart, in_step, out_step);
    if (verbose)
      engine->finish();
    return;
  }

  // CPU version
  for (unsigned input_ichan=0; input_ichan<input->get_nchan(); input_ichan++)
  {
    for (unsigned ipol=0; ipol < npol; ipol++)
    {
      const float* indat = input->get_datptr (input_ichan, ipol);
      // TODO FIX THIS!!!
      float *outdat      = output->get_dattpf ();

      for (uint64_t ipart=0; ipart < npart; ipart++)
      {
        if (input->get_state() == Signal::Nyquist)
          forward->frc1d (nsamp_fft, outdat, indat);
        else
          forward->fcc1d (nsamp_fft, outdat, indat);

        indat  += in_step;
        outdat += out_step;
      }
    }  // input ipol
  } // input_ichan
}


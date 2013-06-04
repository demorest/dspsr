/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SpectralKurtosis.h"
#include "dsp/InputBuffering.h"

#include <errno.h>
#include <assert.h>
#include <string.h>

using namespace std;

dsp::SpectralKurtosis::SpectralKurtosis() : Transformation<TimeSeries,TimeSeries>("SpectralKurtosis", outofplace)
{
  output_tscr = 0;
  tscrunch = 128;
  debugd = 1;

  set_buffering_policy(new InputBuffering(this));
}

dsp::SpectralKurtosis::~SpectralKurtosis () 
{
  if (verbose)
    cerr << "dsp::SpectralKurtosis::SpectralKurtosis~" << endl;
}

bool dsp::SpectralKurtosis::get_order_supported (TimeSeries::Order order) const
{
  if (order == TimeSeries::OrderFPT || order == TimeSeries::OrderTFP)
    return true;
}


void dsp::SpectralKurtosis::set_engine (Engine* _engine)
{
  engine = _engine;
}

/*
 * These are preparations that could be performed once at the start of
 * the data processing
 */
void dsp::SpectralKurtosis::prepare ()
{
  if (verbose)
    cerr << "dsp::SpectralKurtosis::prepare()" << endl;
  prepare_output ();
}

void dsp::SpectralKurtosis::prepare_output ()
{
  if (verbose)
    cerr << "dsp::SpectralKurtosis::prepare_output()" << endl;

  output->copy_configuration ( get_input() );
  output->set_npol (input->get_npol());
  output->set_nchan (input->get_nchan());
  output->set_ndim (1);
  output->set_state (Signal::Analytic);
  output->set_order (TimeSeries::OrderTFP);
  output->set_scale (1.0);

  if (input->get_npol() == 2)
    output->set_state (Signal::PPQQ);
  else
    output->set_state (Signal::Intensity);

  double output_rate = input->get_rate() / tscrunch;
  output->set_rate (output_rate);

  //output->resize( ndat );
}

/* 
 * Return the number of samples required to increment the blocksize
 * such that the SKFB will have a full set of integrations
 */
uint64_t dsp::SpectralKurtosis::get_skfb_inc (uint64_t blocksize)
{

  if (verbose)
    cerr << "dsp::SpectralKurtosis::get_skfb_inc M=" << tscrunch 
         << " blocksize=" << blocksize << endl;

  uint64_t skfb_min = tscrunch;
  uint64_t remainder = blocksize % tscrunch;
  uint64_t increment = skfb_min - remainder;
  
  if (verbose)
    cerr << "dsp::SpectralKurtosis::get_skfb_inc skfb_min=" << skfb_min 
         << " remainder=" << remainder << " inc=" << increment << endl;

  return increment; 

}

void dsp::SpectralKurtosis::set_output_tscr (TimeSeries * _output_tscr)
{
  if (verbose)
    cerr << "dsp::SpectralKurtosis::set_output_tscr()" << endl;
  output_tscr = _output_tscr;
  output_tscr->set_order(TimeSeries::OrderTFP);
}

void dsp::SpectralKurtosis::transformation ()
{
  const uint64_t ndat  = input->get_ndat();
  const unsigned nchan = input->get_nchan();
  const unsigned npol  = input->get_npol();
  const unsigned ndim  = input->get_ndim();

  if (verbose || debugd < 1)
    cerr << "dsp::SpectralKurtosis::transformation input ndat=" << ndat
         << " nchan=" << nchan << " npol=" << npol << " ndim=" << ndim 
         << " tscrunch=" << tscrunch << endl;

  const uint64_t nscrunch = ndat / tscrunch;
  const uint64_t final_sample = nscrunch * tscrunch;

  if (verbose || debugd < 1)
    cerr << "dsp::SpectralKurtosis::transformation input nscrunch=" << nscrunch 
         << " final_sample=" << nscrunch * tscrunch << endl;


  if (has_buffering_policy())
  {
    if (verbose || debugd < 1)
      cerr << "dsp::SpectralKurtosis::transformation setting next_start_sample=" 
           << final_sample << endl;
    get_buffering_policy()->set_next_start (final_sample);
  }

  // ensure output SK estimate timeseries is configured
  prepare_output ();
  output->resize (nscrunch);

  // adjust tscr output
  if (output_tscr && nchan > output_tscr->get_nchan())
  {
    //if (verbose)
      cerr << "dsp::SpectralKurtosis::transformation output_tscr set_nchan(" << nchan << ")" << endl;
    output_tscr->set_ndat (1);
    output_tscr->set_nchan (nchan);
    output_tscr->set_npol (npol);
    output_tscr->set_ndim (1);
    output_tscr->resize (1);

    //if (verbose)
      cerr << "dsp::SpectralKurtosis::transformation S?_tscr.resize(" << nchan*npol << ")" << endl;
    S1_tscr.resize(nchan * npol);
    S2_tscr.resize(nchan * npol);
  }

  // initialise tscr
  if (output_tscr)
  {
    for (unsigned i=0; i<nchan*npol; i++)
    {
      S1_tscr[i]=0;
      S2_tscr[i]=0;
    }
  }
  
  float S1_sum, S2_sum;

  const float M = (float) tscrunch;
  const float M_fac = (M+1) / (M-1);

  float * outdat = output->get_dattfp();

  switch (input->get_order())
  {
    case dsp::TimeSeries::OrderTFP:
    {
      const unsigned int chan_stride = nchan * npol * ndim;
      float * indat;

      for (unsigned iscrunch=0; iscrunch < nscrunch; iscrunch++)
      {
        indat = (float *) input->get_dattfp() + (tscrunch * iscrunch * chan_stride);
        
        for (unsigned ichan=0; ichan<nchan; ichan++)
        {
          for (unsigned ipol=0; ipol < npol; ipol++)
          {
            S1_sum = 0;
            S2_sum = 0;

            // Square Law Detect for S1 + S2
            for (unsigned i=0; i<tscrunch; i++)
            {
              float re = indat[chan_stride*i];
              float im = indat[chan_stride*i+1];
              float sqld = (re * re) + (im * im);
              S1_sum += sqld;
              S2_sum += (sqld * sqld);
            }

            // add the sums to the tscrunch timeseries
            S1_tscr [ichan*npol + ipol] += S1_sum;
            S2_tscr [ichan*npol + ipol] += S2_sum;

            // calculate the SK estimator
            outdat[ichan*npol + ipol] = M_fac * (M * (S2_sum / (S1_sum * S1_sum)) - 1);
            
            indat += ndim;
          }
        }
        outdat += nchan * npol;
      }
      break;
    }

    case dsp::TimeSeries::OrderFPT:
    {
      const unsigned int nfloat = tscrunch * ndim;
      // foreach input channel
      for (unsigned iscrunch=0; iscrunch < nscrunch; iscrunch++)
      {
        for (unsigned ichan=0; ichan<nchan; ichan++)
        {
          for (unsigned ipol=0; ipol < npol; ipol++)
          {
            // input pointer for channel pol
            const float* indat = input->get_datptr (ichan, ipol) + iscrunch * nfloat;

            S1_sum = 0;
            S2_sum = 0;

            // Square Law Detect for S1 + S2
            for (unsigned i=0; i<nfloat; i+=2)
            {
              float sqld = (indat[i] * indat[i]) + (indat[i+1] * indat[i+1]);
              S1_sum += sqld;
              S2_sum += (sqld * sqld);
            }
          
            // add the sums to the tscrunch timeseries
            S1_tscr [ichan*npol + ipol] += S1_sum;
            S2_tscr [ichan*npol + ipol] += S2_sum;

            // calculate the SK estimator
            outdat[ichan*npol + ipol] = M_fac * (M * (S2_sum / (S1_sum * S1_sum)) - 1);
          }
        }
        outdat += nchan * npol;
      }
      break;
    }

    default:
    {
      throw Error (InvalidState, "dsp::SpectralKurtosis::transformation", "unsupported input order");
    }
  }

  // calculate the SK Estimator for the whole block of data
  if (output_tscr)
  {
    float M = (float) (tscrunch * nscrunch);
    float M_fac = (M+1) / (M-1);
    float * outdat = output_tscr->get_dattfp();

    if (debugd < 1)
      cerr << "dsp::SpectralKurtosis::transformation tscr M=" << M <<" M_fac=" << M_fac << endl;
    for (unsigned ichan=0; ichan<nchan; ichan++)
    {
      for (unsigned ipol=0; ipol<npol; ipol++)
      {
        S1_sum = S1_tscr[ichan*npol + ipol];
        S2_sum = S2_tscr[ichan*npol + ipol];
        outdat[ichan*npol + ipol] = M_fac * (M * (S2_sum / (S1_sum * S1_sum)) - 1);
      }
    }
  }

  uint64_t input_sample = input->get_input_sample();
  uint64_t output_sample = input_sample / tscrunch;
  output->set_input_sample (output_sample);

  if (verbose || debugd < 1)
    cerr << "dsp::SpectralKurtosis::transformation input_sample=" << input_sample 
         << ", output_sample=" << output->get_input_sample() << ", ndat=" 
         << output->get_ndat() << endl;

  if (verbose || debugd < 1)
    cerr << "dsp::SpectralKurtosis::transformation done" << endl;
  if (debugd < 1)
    debugd++;

}

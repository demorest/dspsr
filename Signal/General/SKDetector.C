/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SKDetector.h"

#include "dsp/Unpacker.h"
#include "dsp/Input.h"
#include "dsp/SKLimits.h"

#include <assert.h>
#include <iomanip>

using namespace std;

dsp::SKDetector::SKDetector () 
  : Transformation<TimeSeries,BitSeries>("SKDetector",outofplace)
{
  M = 0;
  one_sigma = 0;
  n_std_devs = 3;
  upper_thresh = 0;
  lower_thresh = 0;
  debugd = 1;
  s_chan = 0;
  e_chan = 0;
  ndat_zapped = 0;
  ndat_zapped_skfb = 0;
  ndat_zapped_fscr = 0;
  ndat_zapped_tscr = 0;
  ndat_total = 0;
  disable_fscr = false;
  disable_tscr = false;
  disable_ft = false;
  input_tscr = 0;
  tscr_M = 0;
  tscr_upper = 0;
  tscr_lower = 0;

  unfiltered_hits = 0;
}

dsp::SKDetector::~SKDetector ()
{
  if (verbose)
    cerr << "dsp::SKDetector::~SKDetector()" << endl;

  cerr << "Zapped: " 
       << " total=" << (100 * (float) ndat_zapped / (float) ndat_total) << "\%" 
       << " skfb=" << (100 * (float) ndat_zapped_skfb / (float) ndat_total) << "\%"
       << " tscr=" << (100 * (float) ndat_zapped_tscr / (float) ndat_total) << "\%"
       << " fscr=" << (100 * (float) ndat_zapped_fscr / (float) ndat_total) << "\%"
       << endl;
}

void dsp::SKDetector::set_engine (Engine * _engine)
{
  engine = _engine;
  engine->setup();
}

void dsp::SKDetector::set_thresholds (unsigned _M, unsigned _n_std_devs)
{
  M = _M;
  n_std_devs = _n_std_devs;

  if (verbose)
    cerr << "dsp::SKDetector::set_thresholds SKlimits(" << M << ", " << n_std_devs << ")" << endl;
  dsp::SKLimits limits(M, n_std_devs);
  limits.calc_limits();

  upper_thresh = (float) limits.get_upper_threshold();
  lower_thresh = (float) limits.get_lower_threshold();

  if (verbose)
    cerr << "dsp::SKDetector::set_thresholds M=" << M << " n_std_devs="
         << n_std_devs  << " [" << lower_thresh << " - " << upper_thresh 
         << "]" << endl;
}

void dsp::SKDetector::set_channel_range (unsigned start, unsigned end)
{
  s_chan = start;
  e_chan = end;
}

void dsp::SKDetector::set_options (bool _disable_fscr, bool _disable_tscr,
                                   bool _disable_ft)
{
  disable_fscr = _disable_fscr;
  disable_tscr = _disable_tscr;
  disable_ft   = _disable_ft;
}

void dsp::SKDetector::set_input_tscr (TimeSeries * _input_tscr)
{
  input_tscr = _input_tscr;
}


void dsp::SKDetector::reserve()
{
  output->Observation::operator=(*input);
  output->set_npol (1);
  output->set_ndim (1);
  output->set_nbit (8);
  output->set_nchan (input->get_nchan());
  output->set_input_sample (input->get_input_sample());
  output->resize(input->get_ndat());
}

void dsp::SKDetector::transformation ()
{

  // ensure containers are large enough
  reserve();

  const unsigned nchan = input->get_nchan();
  const unsigned npol  = input->get_npol();
  int64_t ndat         = input->get_ndat();

  if (e_chan == 0)
    e_chan = nchan;

  if (verbose || debugd < 1)
  {
    cerr << "dsp::SKDetector::transformation ndat= " << ndat 
         << " nchan=" << nchan << " nbit=" << input->get_nbit() 
         << " npol=" << npol << " ndim=" << input->get_ndim() << endl;

    cerr << "dsp::SKDetector::transformation OUTPUT ndat="
         << output->get_ndat() << " nchan=" << output->get_nchan()
         << " nbit=" << output->get_nbit() << " npol=" << output->get_npol() 
         << " ndim=" << output->get_ndim() << endl;
  }

  ndat_total += ndat * nchan;

  // reset the output to all 0 (no zapping)
  reset_mask();

  // apply the tscrunches SKFB estiamtes to the mask
  if (!disable_tscr)
    detect_tscr();

  // apply the SKFB estimates to the mask
  if (!disable_ft)
    detect_skfb();

  if (!disable_fscr)
    detect_fscr ();

  count_zapped ();

  if (debugd < 1)
    debugd++;
}

/*
 * Use the tscrunched SK statistic from the SKFB to detect RFI on eah channel
 */
void dsp::SKDetector::detect_tscr ()
{

  if (verbose)
    cerr << "dsp::SKDetector::detect_tscr()" << endl;

  const unsigned nchan   = input->get_nchan();
  int64_t ndat           = input->get_ndat();
  const float * indat    = input_tscr->get_dattfp();
  unsigned char * outdat = 0;

  const unsigned npol    = input->get_npol();
  unsigned zap_chan;
  float V;
  //assert(npol == 2);

  //float V_p0 = 0;
  //float V_p1 = 0;

  if (ndat && (tscr_M != M * ndat))
  {
    tscr_M = M * ndat;

    if (verbose)
      cerr << "dsp::SKDetector::detect_tscr SKlimits(" << tscr_M << ", " << n_std_devs << ")" << endl;

    dsp::SKLimits limits(tscr_M, n_std_devs);
    limits.calc_limits();

    tscr_upper = (float) limits.get_upper_threshold();
    tscr_lower = (float) limits.get_lower_threshold();

    if (verbose)
      cerr << "dsp::SKDetector::detect_tscr M=" << tscr_M << " n_std_devs="
           << n_std_devs  << " [" << tscr_lower << " - " << tscr_upper
           << "]" << endl;
  }

  if (engine)
  {
    engine->detect_tscr(input, input_tscr, output, tscr_upper, tscr_lower);
    return;
  }

  for (uint64_t ichan=s_chan; ichan < e_chan; ichan++)
  {
    zap_chan = 0;
    for (unsigned ipol=0; ipol < npol; ipol++)
    {
      V = indat[ichan*npol + ipol];
      if (V > tscr_upper || V < tscr_lower)
        zap_chan = 1;

    }
    if (zap_chan)
    {
      if (verbose)
        cerr << "dsp::SKDetector::detect_tscr zap V=" << V << ", " 
             << "ichan=" << ichan << endl;
      outdat = output->get_datptr();
      for (unsigned idat=0; idat < ndat; idat++)
      {
        outdat[ichan] = 1;
        ndat_zapped_tscr++;
        outdat += nchan;
      }
    }
  }
}

void dsp::SKDetector::detect_skfb ()
{

  if (verbose)
    cerr << "dsp::SKDetector::detect_skfb()" << endl;

  if (engine)
  {
    engine->detect_ft(input, output, upper_thresh, lower_thresh);
    return;
  }

  const unsigned nchan   = input->get_nchan();
  const unsigned npol    = input->get_npol();
  uint64_t ndat          = input->get_ndat();
  const float * indat    = input->get_dattfp();
  unsigned char * outdat = output->get_datptr();
  float V = 0;
  char zap;

  // compare SK estimator for each pol to expected values
  for (uint64_t idat=0; idat < ndat; idat++)
  {
    // for each channel and pol in the SKFB
    for (unsigned ichan=0; ichan < nchan; ichan++)
    {
      zap = 0;
      for (unsigned ipol=0; ipol < npol; ipol++)
      {
        V = indat[npol*ichan + ipol];
        if (V > upper_thresh || V < lower_thresh)
        {
          zap = 1;
        }
      }
      if (zap)
      {
        outdat[ichan] = 1;

        // only count skfb zapped channels in the in-band region
        if (ichan > s_chan && ichan < e_chan)
          ndat_zapped_skfb++;
      }
    }

    indat += nchan * npol;
    outdat += nchan; 
  }
}

void dsp::SKDetector::reset_mask()
{
  if (engine)
  {
    engine->reset_mask(output);
    return;
  }

  unsigned nchan         = output->get_nchan();
  uint64_t ndat          = output->get_ndat();
  unsigned char * outdat = output->get_datptr();

  for (unsigned ichan=0; ichan < nchan; ichan++)
  {
    for (uint64_t idat=0; idat < ndat; idat++)
    {
      outdat[(idat*nchan) + ichan] = 0;
    }
  }
}



void dsp::SKDetector::count_zapped ()
{
  if (verbose)
    cerr << "dsp::SKDetector::count_zapped hits=" << unfiltered_hits << endl;

  int zapped = 0;

  if (engine)
  {
    zapped = engine->count_mask (output);
    ndat_zapped += zapped;
    return;
  }

  unsigned npol          = input->get_npol();
  const float * indat    = input->get_dattfp();

  unsigned nchan         = output->get_nchan();
  uint64_t ndat          = output->get_ndat();
  unsigned char * outdat = output->get_datptr();

  assert (ndat == input->get_ndat());

  if (unfiltered_hits == 0)
    {
      filtered_sum.resize (npol * nchan);
      std::fill (filtered_sum.begin(), filtered_sum.end(), 0);

      filtered_hits.resize (nchan);
      std::fill (filtered_hits.begin(), filtered_hits.end(), 0);

      unfiltered_sum.resize (npol * nchan);
      std::fill (unfiltered_sum.begin(), unfiltered_sum.end(), 0);
    }

  for (uint64_t idat=0; idat < ndat; idat++)
  {
    // number of SK idats (same for each channel)
    unfiltered_hits ++;

    for (unsigned ichan=s_chan; ichan < e_chan; ichan++)
    {
      uint64_t index = (idat*nchan + ichan) * npol;
      unsigned outdex = ichan * npol;

      // sum of all SK values 
      unfiltered_sum[outdex] += indat[index];
      if (npol == 2)
        unfiltered_sum[outdex+1] += indat[index+1];
  
      if (outdat[(idat*nchan) + ichan] == 1)
      {
        ndat_zapped ++;
        continue;
      }

      filtered_sum[outdex] += indat[index];
      if (npol == 2)
        filtered_sum[outdex+1] += indat[index+1];

      filtered_hits[ichan] ++;
    }
  }
}

void dsp::SKDetector::detect_fscr ()
{ 
  if (verbose)
    cerr << "dsp::SKDetector::detect_fscr()" << endl;

  float _M = (float) M;
  float mu2 = (4 * _M * _M) / ((_M-1) * (_M + 2) * (_M + 3));
  unsigned nchan       = input->get_nchan();

  if (engine)
  {
    float one_sigma_idat   = sqrt(mu2 / float((e_chan-s_chan)+1));
    const float upper = 1 + ((1+n_std_devs) * one_sigma_idat);
    const float lower = 1 - ((1+n_std_devs) * one_sigma_idat);
    engine->detect_fscr(input, output, lower, upper, s_chan, e_chan);
    return;
  }

  const unsigned npol  = input->get_npol();
  const uint64_t ndat  = input->get_ndat();

  const float * indat  = input->get_dattfp();
  unsigned char * outdat = output->get_datptr();

  float sk_avg;
  unsigned sk_avg_cnt = 0;
  
  unsigned zap_idat;
  uint64_t nzap = 0;

  // foreach SK integration
  for (uint64_t idat=0; idat < ndat; idat++)
  {
    zap_idat = 0;
    for (unsigned ipol=0; ipol < npol; ipol++)
    {
      sk_avg = 0;
      sk_avg_cnt = 0;

      for (unsigned ichan=s_chan; ichan < e_chan; ichan++)
      {
        if (outdat[ichan] == 0)
        {
          sk_avg += indat[ichan*npol + ipol];
          sk_avg_cnt++;
        }
      }

      if (sk_avg_cnt > 0)
      {
        sk_avg /= (float) sk_avg_cnt;

        float one_sigma_idat = sqrt(mu2 / (float) sk_avg_cnt);
        float avg_upper_thresh = 1 + ((1+n_std_devs) * one_sigma_idat);
        float avg_lower_thresh = 1 - ((1+n_std_devs) * one_sigma_idat);
        if ((sk_avg > avg_upper_thresh) || (sk_avg < avg_lower_thresh))
        {
          if (verbose)
            cerr << "Zapping idat=" << idat << " ipol=" << ipol << " sk_avg=" << sk_avg
                 << " [" << avg_lower_thresh << " - " << avg_upper_thresh
                 << "] cnt=" << sk_avg_cnt << endl;
          zap_idat = 1;
        }
      }
    }

    if (zap_idat)
    {
      for (unsigned ichan=0; ichan<nchan; ichan++)
      {
        outdat[ichan] = 1;
      }
      //ndat_zapped_fscr += sk_avg_cnt;
      ndat_zapped_fscr += nchan;
      nzap += nchan;
    }

    indat += nchan * npol;
    outdat += nchan;
  }
  //cerr << "dsp::SKDetector::detect_fscr ZAP=" << nzap << endl;
}


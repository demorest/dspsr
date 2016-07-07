/***************************************************************************
 *
 *   Copyright (C) 2016 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SpectralKurtosis.h"
#include "dsp/InputBuffering.h"
#include "dsp/SKLimits.h"

#include <errno.h>
#include <assert.h>
#include <string.h>

using namespace std;

dsp::SpectralKurtosis::SpectralKurtosis() : Transformation<TimeSeries,TimeSeries>("SpectralKurtosis", outofplace)
{
  M = 128;
  debugd = 1;

  estimates = new TimeSeries;
  estimates_tscr = new TimeSeries;
  zapmask = new BitSeries;

  // SK Detector
  std_devs = 3;
  channels.resize(2);
  npart_total = 0;
  thresholds.resize(2);
  thresholds_tscr.resize(2);
  zap_counts.resize(4);
  detection_flags.resize(3);
  std::fill (detection_flags.begin(), detection_flags.end(), false);
  detection_flags.resize(3);
  M_tscr = 0;

  unfiltered_hits = 0;

  prepared = false;

  set_buffering_policy(new InputBuffering(this));
}

dsp::SpectralKurtosis::~SpectralKurtosis ()
{
  if (verbose)
    cerr << "dsp::SpectralKurtosis::SpectralKurtosis~" << endl;

  float percent_all = 0;
  float percent_skfb = 0;
  float percent_tscr = 0;
  float percent_fscr = 0;

  if (npart_total)
  {
    percent_all  = (100 * (float) zap_counts[ZAP_ALL]  / (float) npart_total);
    percent_skfb = (100 * (float) zap_counts[ZAP_SKFB] / (float) npart_total);
    percent_tscr = (100 * (float) zap_counts[ZAP_TSCR] / (float) npart_total);
    percent_fscr = (100 * (float) zap_counts[ZAP_FSCR] / (float) npart_total);
  }

  cerr << "Zapped: " 
       << " total=" << percent_all <<  "\%" << " skfb=" << percent_skfb << "\%"
       << " tscr=" << percent_tscr << "\%" << " fscr=" << percent_fscr << "\%"
       << endl;

  delete estimates;
  delete estimates_tscr;
  delete zapmask;
}

bool dsp::SpectralKurtosis::get_order_supported (TimeSeries::Order order) const
{
  if (order == TimeSeries::OrderFPT || order == TimeSeries::OrderTFP)
    return true;
}


void dsp::SpectralKurtosis::set_engine (Engine* _engine)
{
  if (verbose)
    cerr << "dsp::SpectralKurtosis::set_engine()" << endl;
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

  nchan = input->get_nchan();
  npol = input->get_npol();
  ndim = input->get_ndim();

  Memory * memory = const_cast<Memory *>(input->get_memory());
  estimates->set_memory (memory);
  estimates_tscr->set_memory (memory);
  zapmask->set_memory (memory);

  if (has_buffering_policy())
  {
    get_buffering_policy()->set_minimum_samples (M);
  }

  if (engine)
  {
    engine->setup ();
  }
  else
  {
    if (!detection_flags[1])
    {
      S1_tscr.resize(nchan * npol);
      S2_tscr.resize(nchan * npol);
    }
  }

  // ensure output containers are configured correctly
  prepare_output ();

  prepared = true;
}

/*! ensure output parameters are configured correctly */
void dsp::SpectralKurtosis::prepare_output ()
{
  if (verbose)
    cerr << "dsp::SpectralKurtosis::prepare_output()" << endl;
    
  double mask_rate = input->get_rate() / M;

  estimates->copy_configuration (get_input());
  estimates->set_ndim (1);                      // SK estimates have only single dimension
  estimates->set_order (TimeSeries::OrderTFP);  // stored in TFP order
  estimates->set_scale (1.0);                   // no scaling
  estimates->set_rate (mask_rate);              // rate is /= M

  if (input->get_npol() == 2)
    estimates->set_state (Signal::PPQQ);
  else
    estimates->set_state (Signal::Intensity);

  double tscrunch_mask_rate = mask_rate;
  if (npart > 0)
    tscrunch_mask_rate /= npart;

  // tscrunched estimates have same configuration, except number of samples
  estimates_tscr->copy_configuration (estimates);
  estimates_tscr->set_order (TimeSeries::OrderTFP);  // stored in TFP order
  estimates_tscr->set_rate (tscrunch_mask_rate);

  // zap mask has same configuration as estimates with following changes
  zapmask->copy_configuration (estimates);
  zapmask->set_nbit (8);
  zapmask->set_npol (1);

  // configure output timeseries (out-of-place) to match input
  output->copy_configuration (get_input()); 
  output->set_input_sample (input->get_input_sample ());
}

/* ensure containers have correct dynamic size */ 
void dsp::SpectralKurtosis::reserve ()
{
  if (verbose)
    cerr << "dsp::SpectralKurtosis::reserve()" << endl;

  const uint64_t ndat  = input->get_ndat();
  npart = ndat / M;
  output_ndat = npart * M;

  if (verbose)
    cerr << "dsp::SpectralKurtosis::reserve input_ndat=" << ndat 
         << " npart=" << npart << " output_ndat=" << output_ndat << endl;

  // use resize since out of place operation
  estimates->resize (npart);
  estimates_tscr->resize (npart > 0); // 1 if npart != 0
  zapmask->resize (npart);
  output->resize (output_ndat);
}

/* call set of transformations */
void dsp::SpectralKurtosis::transformation ()
{
  if (!prepared)
    prepare();

  const uint64_t ndat  = input->get_ndat();
  if (verbose || debugd < 1)
    cerr << "dsp::SpectralKurtosis::transformation input ndat=" << ndat
         << " tscrunch=" << M << endl;

  npart = ndat / M;
  output_ndat = npart * M;

  if (verbose || debugd < 1)
    cerr << "dsp::SpectralKurtosis::transformation input npart=" << npart
         << " output_ndat=" << output_ndat << endl;

  if (has_buffering_policy())
  {
    if (verbose || debugd < 1)
      cerr << "dsp::SpectralKurtosis::transformation setting next_start_sample="
           << output_ndat << endl;
    get_buffering_policy()->set_next_start (output_ndat);
  }

  prepare_output ();

  // ensure output containers are sized correctly
  reserve ();
  
  if ((ndat == 0) || (npart == 0))
    return;

  // perform SK functions
  compute ();
  detect ();
  mask ();
  //insertsk();
}

void dsp::SpectralKurtosis::compute ()
{
  if (verbose)
    cerr << "dsp::SpectralKurtosis::compute" << endl;

  if (engine)
  {
    engine->compute (input, estimates, estimates_tscr, M);
  }
  else
  {
    // initialise tscr
    if (!detection_flags[1])
    {
      std::fill(S1_tscr.begin(), S1_tscr.end(), 0);
      std::fill(S2_tscr.begin(), S2_tscr.end(), 0);
    }

    float S1_sum, S2_sum;
    const float M_fac = (M+1) / (M-1);
    float * outdat = estimates->get_dattfp();

    switch (input->get_order())
    {
      case dsp::TimeSeries::OrderTFP:
      {
        const unsigned int chan_stride = nchan * npol * ndim;
        float * indat;

        for (unsigned ipart=0; ipart < npart; ipart++)
        {
          indat = (float *) input->get_dattfp() + (M * ipart * chan_stride);

          for (unsigned ichan=0; ichan<nchan; ichan++)
          {
            for (unsigned ipol=0; ipol < npol; ipol++)
            {
              S1_sum = 0;
              S2_sum = 0;

              // Square Law Detect for S1 + S2
              for (unsigned i=0; i<M; i++)
              {
                float re = indat[chan_stride*i];
                float im = indat[chan_stride*i+1];
                float sqld = (re * re) + (im * im);
                S1_sum += sqld;
                S2_sum += (sqld * sqld);
              }

              // add the sums to the M timeseries
              S1_tscr [ichan*npol + ipol] += S1_sum;
              S2_tscr [ichan*npol + ipol] += S2_sum;

              // calculate the SK estimator
              if (S1_sum == 0)
                outdat[ichan*npol + ipol] = 0;
              else
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
        const unsigned int nfloat = M * ndim;
        // foreach input channel
        for (unsigned ipart=0; ipart < npart; ipart++)
        {
          for (unsigned ichan=0; ichan<nchan; ichan++)
          {
            for (unsigned ipol=0; ipol < npol; ipol++)
            {
              // input pointer for channel pol
              const float* indat = input->get_datptr (ichan, ipol) + ipart * nfloat;

              S1_sum = 0;
              S2_sum = 0;

              // Square Law Detect for S1 + S2
              for (unsigned i=0; i<nfloat; i+=2)
              {
                float sqld = (indat[i] * indat[i]) + (indat[i+1] * indat[i+1]);
                S1_sum += sqld;
                S2_sum += (sqld * sqld);
              }

              // add the sums to the M timeseries
              if (!detection_flags[1])
              {
                S1_tscr [ichan*npol + ipol] += S1_sum;
                S2_tscr [ichan*npol + ipol] += S2_sum;
              }

              // calculate the SK estimator
              if (S1_sum == 0)
                outdat[ichan*npol + ipol] = 0;
              else
                outdat[ichan*npol + ipol] = M_fac * (M * (S2_sum / (S1_sum * S1_sum)) - 1);
            }
          }
          outdat += nchan * npol;
        }
        break;
      }

      default:
      {
        throw Error (InvalidState, "dsp::SpectralKurtosis::compute", "unsupported input order");
      }
    }

    // calculate the SK Estimator for the whole block of data
    if (!detection_flags[1])
    {
      float M_t = (float) (M * npart);
      float M_fac = (M_t+1) / (M_t-1);
      float * outdat = estimates_tscr->get_dattfp();
      if (verbose || debugd < 1)
        cerr << "dsp::SpectralKurtosis::compute tscr M=" << M_t <<" M_fac=" << M_fac << endl;
      for (unsigned ichan=0; ichan<nchan; ichan++)
      {
        for (unsigned ipol=0; ipol<npol; ipol++)
        {
          S1_sum = S1_tscr[ichan*npol + ipol];
          S2_sum = S2_tscr[ichan*npol + ipol];
          if (S1_sum == 0)
            outdat[ichan*npol + ipol] = 0;
          else
            outdat[ichan*npol + ipol] = M_fac * (M_t * (S2_sum / (S1_sum * S1_sum)) - 1);
        }
      }
    }
  }

  if (verbose || debugd < 1)
    cerr << "dsp::SpectralKurtosis::compute done" << endl;
  if (debugd < 1)
    debugd++;
}

void dsp::SpectralKurtosis::set_thresholds (unsigned _M, unsigned _std_devs)
{
  M = _M;
  std_devs = _std_devs;

  if (verbose)
    cerr << "dsp::SpectralKurtosis::set_thresholds SKlimits(" << M << ", " << std_devs << ")" << endl;
  dsp::SKLimits limits(M, std_devs);
  limits.calc_limits();

  thresholds[0] = (float) limits.get_lower_threshold();
  thresholds[1] = (float) limits.get_upper_threshold();

  if (verbose)
    cerr << "dsp::SpectralKurtosis::set_thresholds M=" << M << " std_devs="
         << std_devs  << " [" << thresholds[0] << " - " << thresholds[1]
         << "]" << endl;
}

void dsp::SpectralKurtosis::set_channel_range (unsigned start, unsigned end)
{
  channels[0] = start;
  channels[1] = end;
}

void dsp::SpectralKurtosis::set_options (bool _disable_fscr, 
    bool _disable_tscr, bool _disable_ft)
{
  detection_flags[0] = _disable_fscr;
  detection_flags[1] = _disable_tscr;
  detection_flags[2] = _disable_ft;
}

void dsp::SpectralKurtosis::detect ()
{
  if (verbose)
    cerr << "dsp::SpectralKurtosis::detect" << endl;

  // if no end channel was specified, do them all
  if (channels[1] == 0)
    channels[1] = nchan;

  if (verbose || debugd < 1)
  {
    cerr << "dsp::SpectralKurtosis::detect npart= " << npart  
         << " nchan=" << nchan << " nbit=" << input->get_nbit() 
         << " npol=" << npol << " ndim=" << ndim << endl;

    cerr << "dsp::SpectralKurtosis::detect OUTPUT ndat="
         << zapmask->get_ndat() << " nchan=" << zapmask->get_nchan()
         << " nbit=" << zapmask->get_nbit() << " npol=" << zapmask->get_npol() 
         << " ndim=" << zapmask->get_ndim() << endl;
  }

  npart_total += (npart * nchan);

  // reset the mask to all 0 (no zapping)
  reset_mask();

  // apply the tscrunches SKFB estiamtes to the mask
  if (!detection_flags[1])
    detect_tscr ();

  // apply the SKFB estimates to the mask
  if (!detection_flags[2])
    detect_skfb ();

  if (!detection_flags[0])
    detect_fscr ();

  count_zapped ();

  if (debugd < 1)
    debugd++;
}

/*
 * Use the tscrunched SK statistic from the SKFB to detect RFI on eah channel
 */
void dsp::SpectralKurtosis::detect_tscr ()
{
  if (verbose)
    cerr << "dsp::SpectralKurtosis::detect_tscr(" << npart << ")" << endl;

  const float * indat    = estimates_tscr->get_dattfp();
  unsigned char * outdat = 0;
  unsigned zap_chan;
  float V;

  if (npart && (M_tscr != M * npart))
  {
    M_tscr = (float) (M * npart);

    if (verbose)
      cerr << "dsp::SpectralKurtosis::detect_tscr SKlimits(" << M_tscr << ", " << std_devs << ")" << endl;

    dsp::SKLimits limits(M_tscr, std_devs);
    limits.calc_limits();

    thresholds_tscr[0] = (float) limits.get_lower_threshold();
    thresholds_tscr[1] = (float) limits.get_upper_threshold();

    if (verbose)
      cerr << "dsp::SpectralKurtosis::detect_tscr M=" << M_tscr << " std_devs="
           << std_devs  << " [" << thresholds_tscr[0] << " - " << thresholds_tscr[1]
           << "]" << endl;
  }

  if (engine)
  {
    engine->detect_tscr (estimates, estimates_tscr, zapmask, thresholds_tscr[1], thresholds_tscr[0]);
    return;
  }

  for (uint64_t ichan=channels[0]; ichan < channels[1]; ichan++)
  {
    zap_chan = 0;
    for (unsigned ipol=0; ipol < npol; ipol++)
    {
      V = indat[ichan*npol + ipol];
      if (V > thresholds_tscr[1] || V < thresholds_tscr[0])
        zap_chan = 1;
    }

    if (zap_chan)
    {
      if (verbose)
        cerr << "dsp::SpectralKurtosis::detect_tscr zap V=" << V << ", " 
             << "ichan=" << ichan << endl;
      outdat = zapmask->get_datptr();
      for (unsigned ipart=0; ipart < npart; ipart++)
      {
        outdat[ichan] = 1;
        zap_counts[ZAP_TSCR]++;
        outdat += nchan;
      }
    }
  }
}

void dsp::SpectralKurtosis::detect_skfb ()
{
  if (verbose)
    cerr << "dsp::SpectralKurtosis::detect_skfb(" << npart << ")" << endl;

  if (engine)
  {
    engine->detect_ft (estimates, zapmask, thresholds[1], thresholds[0]);
    return;
  }

  const float * indat    = estimates->get_dattfp();
  unsigned char * outdat = zapmask->get_datptr();
  float V = 0;
  char zap;

  // compare SK estimator for each pol to expected values
  for (uint64_t ipart=0; ipart < npart; ipart++)
  {
    // for each channel and pol in the SKFB
    for (unsigned ichan=0; ichan < nchan; ichan++)
    {
      zap = 0;
      for (unsigned ipol=0; ipol < npol; ipol++)
      {
        V = indat[npol*ichan + ipol];
        if (V > thresholds[1] || V < thresholds[0])
        {
          zap = 1;
        }
      }
      if (zap)
      {
        outdat[ichan] = 1;

        // only count skfb zapped channels in the in-band region
        if (ichan > channels[0] && ichan < channels[1])
          zap_counts[ZAP_SKFB]++;
      }
    }

    indat += nchan * npol;
    outdat += nchan; 
  }
}

void dsp::SpectralKurtosis::reset_mask ()
{
  if (engine)
  {
    engine->reset_mask (zapmask);
    return;
  }

  unsigned char * outdat = zapmask->get_datptr();

  for (unsigned ichan=0; ichan < nchan; ichan++)
  {
    for (uint64_t ipart=0; ipart < npart; ipart++)
    {
      outdat[(ipart*nchan) + ichan] = 0;
    }
  }
}

void dsp::SpectralKurtosis::count_zapped ()
{
  if (verbose)
    cerr << "dsp::SpectralKurtosis::count_zapped hits=" << unfiltered_hits << endl;

  int zapped = 0;

  const float * indat;
  unsigned char * outdat;

  if (engine)
  {
    int zapped = engine->count_mask (zapmask);
    indat = engine->get_estimates (estimates);
    outdat = engine->get_zapmask(zapmask);
    zap_counts[ZAP_ALL] += zapped;
  }
  else
  {
    indat    = estimates->get_dattfp();
    outdat = zapmask->get_datptr();
  }

  assert (npart == estimates->get_ndat());
  if (unfiltered_hits == 0)
  {
    filtered_sum.resize (npol * nchan);
    std::fill (filtered_sum.begin(), filtered_sum.end(), 0);

    filtered_hits.resize (nchan);
    std::fill (filtered_hits.begin(), filtered_hits.end(), 0);

    unfiltered_sum.resize (npol * nchan);
    std::fill (unfiltered_sum.begin(), unfiltered_sum.end(), 0);
  }

  for (uint64_t ipart=0; ipart < npart; ipart++)
  {
    unfiltered_hits ++;

    for (unsigned ichan=channels[0]; ichan < channels[1]; ichan++)
    {
      uint64_t index = (ipart*nchan + ichan) * npol;
      unsigned outdex = ichan * npol;

      unfiltered_sum[outdex] += indat[index];
      if (npol == 2)
        unfiltered_sum[outdex+1] += indat[index+1];
  
      if (outdat[(ipart*nchan) + ichan] == 1)
      {
        zap_counts[ZAP_ALL] ++;
        continue;
      }

      filtered_sum[outdex] += indat[index];
      if (npol == 2)
        filtered_sum[outdex+1] += indat[index+1];

      filtered_hits[ichan] ++;
    }
  }
}

void dsp::SpectralKurtosis::detect_fscr ()
{
  if (verbose)
    cerr << "dsp::SpectralKurtosis::detect_fscr()" << endl;

  float _M = (float) M;
  float mu2 = (4 * _M * _M) / ((_M-1) * (_M + 2) * (_M + 3));

  if (engine)
  {
    float one_sigma_idat   = sqrt(mu2 / (float) nchan);
    const float upper = 1 + ((1+std_devs) * one_sigma_idat);
    const float lower = 1 - ((1+std_devs) * one_sigma_idat);
    engine->detect_fscr (estimates, zapmask, lower, upper, channels[0], channels[1]);
    return;
  }

  const uint64_t ndat  = estimates->get_ndat();

  const float * indat  = estimates->get_dattfp();
  unsigned char * outdat = zapmask->get_datptr();

  float sk_avg;
  unsigned sk_avg_cnt = 0;
  
  unsigned zap_ipart;
  uint64_t nzap = 0;

  // foreach SK integration
  for (uint64_t ipart=0; ipart < npart; ipart++)
  {
    zap_ipart = 0;
    for (unsigned ipol=0; ipol < npol; ipol++)
    {
      sk_avg = 0;
      sk_avg_cnt = 0;

      for (unsigned ichan=channels[0]; ichan < channels[1]; ichan++)
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
        float avg_upper_thresh = 1 + ((1+std_devs) * one_sigma_idat);
        float avg_lower_thresh = 1 - ((1+std_devs) * one_sigma_idat);
        if ((sk_avg > avg_upper_thresh) || (sk_avg < avg_lower_thresh))
        {
          if (verbose)
            cerr << "Zapping ipart=" << ipart << " ipol=" << ipol << " sk_avg=" << sk_avg
                 << " [" << avg_lower_thresh << " - " << avg_upper_thresh
                 << "] cnt=" << sk_avg_cnt << endl;
          zap_ipart = 1;
        }
      }
    }

    if (zap_ipart)
    {
      for (unsigned ichan=0; ichan<nchan; ichan++)
      {
        outdat[ichan] = 1;
      }
      zap_counts[ZAP_FSCR] += nchan;
      nzap += nchan;
    }

    indat += nchan * npol;
    outdat += nchan;
  }
  //cerr << "dsp::SpectralKurtosis::detect_fscr ZAP=" << nzap << endl;
}


//! Perform the transformation on the input time series
void dsp::SpectralKurtosis::mask ()
{
  // indicate the output timeseries contains zeroed data
  output->set_zeroed_data (true);

  // resize the output to ensure the hits array is reallocated
  if (engine)
  {
    if (verbose)
      cerr << "dsp::SpectralKurtosis::transformation output->resize(" << output->get_ndat() << ")" << endl;
    output->resize (output->get_ndat());
  }

  // get base pointer to mask bitseries
  unsigned char * mask = zapmask->get_datptr ();

  if (engine)
  {
    if (verbose)
      cerr << "dsp::SpectralKurtosis::transformation engine->setup(" << nchan << ")" << endl;
    engine->mask (zapmask, input, output, M);
  }
  else
  {
    // mask is a TFP ordered bit series, output is FTP order Timeseries
    const unsigned nfloat = M * ndim;      
    for (unsigned ichan=0; ichan < nchan; ichan++)
    {
      for (unsigned ipol=0; ipol < npol; ipol++)
      {
        const float * indat  = input->get_datptr(ichan, ipol);
        float * outdat = output->get_datptr(ichan, ipol);
        for (uint64_t ipart=0; ipart < npart; ipart++)
        {
          if (mask[ipart*nchan+ichan])
          {
            for (unsigned j=0; j<nfloat; j++)
              outdat[j] = 0;
          }
          else
          {
            for (unsigned j=0; j<nfloat; j++)
              outdat[j] = indat[j];
          }

          indat += nfloat;
          outdat += nfloat;
        }
      }
    }
  }

  if (debugd < 1)
    debugd++;
}

//! 
void dsp::SpectralKurtosis::insertsk ()
{
  if (engine)
    engine->insertsk (estimates, output, M);
}

